// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.IO;
using TensorSharp.Runtime.HuggingFace;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The <c>-hf org/repo[:quant]</c> resolver against a synthetic Hugging Face hub
/// cache. The layout mirrors what the official client writes
/// (models--org--repo/refs/main + snapshots/&lt;commit&gt;/symlinks), because the
/// resolver's whole job is reading that layout faithfully — quant-tag matching,
/// the Q4_K_M → Q8_0 → first-file fallback, split-shard-1 selection, the
/// missing-refs single-snapshot fallback, and error messages that name what IS
/// cached. File contents are irrelevant here: resolution is a directory walk,
/// and the loader is what would read the bytes.
///
/// Both this class and the server-level <c>-hf</c> tests set HF_HUB_CACHE, so
/// they share a non-parallelizing collection — a process-wide environment
/// variable must not be written from two classes racing each other.
/// </summary>
[CollectionDefinition("Hugging Face cache environment", DisableParallelization = true)]
public sealed class HfCacheCollection
{
}

[Collection("Hugging Face cache environment")]
public sealed class HfCacheResolverTests : IDisposable
{
    private readonly string _root;

    public HfCacheResolverTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "ts-hfcache-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { /* best effort */ }
    }

    private const string Sha = "0123456789abcdef0123456789abcdef01234567";

    /// <summary>Create <c>models--acme--&lt;name&gt;</c> with a refs/main commit and the
    /// listed files (paths may include subdirectories) inside its snapshot.</summary>
    private string CacheRepo(string repoName, params string[] snapshotFiles)
    {
        string repoDir = Path.Combine(_root, "models--acme--" + repoName);
        string snapshotDir = Path.Combine(repoDir, "snapshots", Sha);
        Directory.CreateDirectory(snapshotDir);
        Directory.CreateDirectory(Path.Combine(repoDir, "blobs"));
        Directory.CreateDirectory(Path.Combine(repoDir, "refs"));
        File.WriteAllText(Path.Combine(repoDir, "refs", "main"), Sha + "\n");
        foreach (string file in snapshotFiles)
        {
            string path = Path.Combine(snapshotDir, file.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, "stub");
        }
        return repoDir;
    }

    private static string Spec(string repo) => "acme/" + repo;

    // ---- spec parsing ----

    [Theory]
    [InlineData("unsloth/Qwen3.8-27B-GGUF", "unsloth/Qwen3.8-27B-GGUF", null)]
    [InlineData("unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS", "unsloth/Qwen3.8-27B-GGUF", "UD-IQ3_XXS")]
    [InlineData("org/repo:q8_0", "org/repo", "q8_0")]
    public void TryParseSpec_AcceptsRepoWithAndWithoutTag(string value, string expectedRepo, string? expectedTag)
    {
        Assert.True(HfCacheResolver.TryParseSpec(value, out string repoId, out string? tag));
        Assert.Equal(expectedRepo, repoId);
        Assert.Equal(expectedTag, tag);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("just-a-name")]              // no slash
    [InlineData("org/")]                     // empty repo
    [InlineData("/repo")]                    // empty org
    [InlineData("a/b/c")]                    // two slashes
    [InlineData("a/b:")]                     // empty tag
    [InlineData("a/b/c:tag")]                // two slashes with a tag
    [InlineData("C:/models/foo.gguf")]       // a Windows drive letter is a path, not a spec
    [InlineData("org/..%2Fescape")]          // percent-encoding is not a repo character
    [InlineData("org/repo:bad tag")]         // spaces are not repo characters
    public void TryParseSpec_RejectsNonSpecs(string? value)
    {
        Assert.False(HfCacheResolver.TryParseSpec(value, out _, out _));
        Assert.False(HfCacheResolver.LooksLikeHfSpec(value));
    }

    // ---- resolution ----

    [Fact]
    public void ResolveModelPath_ExplicitTag_MatchesCaseInsensitivelyAndPicksThatFile()
    {
        CacheRepo("ModelGGUF", "Model-UD-IQ3_XXS.gguf", "Model-Q8_0.gguf");

        string path = HfCacheResolver.ResolveModelPath(Spec("ModelGGUF:UD-IQ3_XXS"), cacheRoot: _root);

        Assert.EndsWith("Model-UD-IQ3_XXS.gguf", path, StringComparison.Ordinal);
        Assert.True(File.Exists(path));
    }

    [Fact]
    public void ResolveModelPath_WithoutTag_PrefersQ4KmThenQ8_0ThenAnyModelFile()
    {
        CacheRepo("ModelGGUF", "Model-Q8_0.gguf", "Model-Q6_K.gguf");
        Assert.EndsWith("Model-Q8_0.gguf",
            HfCacheResolver.ResolveModelPath(Spec("ModelGGUF"), cacheRoot: _root), StringComparison.Ordinal);

        CacheRepo("Model2GGUF", "Model2-Q6_K.gguf", "Model2-Q4_K_M.gguf");
        Assert.EndsWith("Model2-Q4_K_M.gguf",
            HfCacheResolver.ResolveModelPath(Spec("Model2GGUF"), cacheRoot: _root), StringComparison.Ordinal);

        CacheRepo("Model3GGUF", "Model3-F16.gguf");
        Assert.EndsWith("Model3-F16.gguf",
            HfCacheResolver.ResolveModelPath(Spec("Model3GGUF"), cacheRoot: _root), StringComparison.Ordinal);
    }

    [Fact]
    public void ResolveModelPath_SplitShard_ResolvesToShardOne()
    {
        CacheRepo("ShardedGGUF",
            "Model-Q8_0-00001-of-00003.gguf",
            "Model-Q8_0-00002-of-00003.gguf",
            "Model-Q8_0-00003-of-00003.gguf");

        string path = HfCacheResolver.ResolveModelPath(Spec("ShardedGGUF:Q8_0"), cacheRoot: _root);

        Assert.EndsWith("Model-Q8_0-00001-of-00003.gguf", path, StringComparison.Ordinal);
    }

    [Fact]
    public void ResolveModelPath_SkipsProjectorAndImatrixFiles()
    {
        CacheRepo("MultimodalGGUF",
            "mmproj-Model-Q8_0.gguf",
            "Model-imatrix-Q8_0.gguf",
            "Model-Q8_0.gguf");

        string path = HfCacheResolver.ResolveModelPath(Spec("MultimodalGGUF:Q8_0"), cacheRoot: _root);

        Assert.EndsWith("Model-Q8_0.gguf", path, StringComparison.Ordinal);
    }

    [Fact]
    public void ResolveModelPath_ByDefault_ExcludesDraftSidecars()
    {
        CacheRepo("WithDraftGGUF",
            "Model-Q8_0.gguf",
            "Model-dspark-Q8_0.gguf",
            "Model-mtp-Q8_0.gguf");

        string path = HfCacheResolver.ResolveModelPath(Spec("WithDraftGGUF:Q8_0"), cacheRoot: _root);

        Assert.EndsWith("Model-Q8_0.gguf", path, StringComparison.Ordinal);

        // The --draft-model spelling resolves through the same cache but treats the
        // draft markers as eligible picks — a standalone drafter is exactly what that
        // flag names, and a draft-only repo must resolve.
        CacheRepo("DraftOnlyGGUF", "Draft-dspark-Q8_0.gguf");
        string draft = HfCacheResolver.ResolvePathOrSpec(Spec("DraftOnlyGGUF"), _root);
        Assert.EndsWith("Draft-dspark-Q8_0.gguf", draft, StringComparison.Ordinal);
    }

    [Fact]
    public void ResolvePathOrSpec_ExistingFileWinsOverSpecShape()
    {
        // A real file whose name happens to look like a spec keeps meaning a file —
        // overloading a flag must not break anyone who already passes a path.
        string lookalike = Path.Combine(_root, "acme", "repo.gguf");
        Directory.CreateDirectory(Path.GetDirectoryName(lookalike)!);
        File.WriteAllText(lookalike, "stub");

        Assert.Equal(lookalike, HfCacheResolver.ResolvePathOrSpec(lookalike, _root));
    }

    [Fact]
    public void ResolvePathOrSpec_PlainPathFallsThroughUnchanged()
    {
        const string path = "/models/some-model-Q8_0.gguf";
        Assert.Equal(path, HfCacheResolver.ResolvePathOrSpec(path, _root));
    }

    [Fact]
    public void ResolveModelPath_ExplicitFile_MatchesRepoRelativePath()
    {
        CacheRepo("NestedGGUF", "nested/dir/Model-Q8_0.gguf", "Model-F16.gguf");

        string path = HfCacheResolver.ResolveModelPath(
            Spec("NestedGGUF"), explicitFile: "nested/dir/Model-Q8_0.gguf", cacheRoot: _root);

        Assert.EndsWith(Path.Combine("nested", "dir", "Model-Q8_0.gguf"), path, StringComparison.Ordinal);
    }

    // ---- error surface ----

    [Fact]
    public void ResolveModelPath_MissingRepo_NamesTheRepoTheRootAndTheDownloadCommand()
    {
        CacheRepo("OtherGGUF", "Other-F16.gguf");

        ArgumentException ex = Assert.Throws<ArgumentException>(() =>
            HfCacheResolver.ResolveModelPath(Spec("MissingGGUF"), cacheRoot: _root));

        Assert.Contains("acme/MissingGGUF", ex.Message, StringComparison.Ordinal);
        Assert.Contains(_root, ex.Message, StringComparison.Ordinal);
        Assert.Contains("hf download acme/MissingGGUF", ex.Message, StringComparison.Ordinal);
        // And it points at what IS cached, so a typo is diagnosed in one read.
        Assert.Contains("acme/OtherGGUF", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ResolveModelPath_MissingQuant_ListsTheAvailableGgufs()
    {
        CacheRepo("ModelGGUF", "Model-F16.gguf", "Model-Q8_0.gguf");

        ArgumentException ex = Assert.Throws<ArgumentException>(() =>
            HfCacheResolver.ResolveModelPath(Spec("ModelGGUF:UD-IQ3_XXS"), cacheRoot: _root));

        Assert.Contains("UD-IQ3_XXS", ex.Message, StringComparison.Ordinal);
        Assert.Contains("Model-F16.gguf", ex.Message, StringComparison.Ordinal);
        Assert.Contains("Model-Q8_0.gguf", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ResolveModelPath_MissingRefs_FallsBackToASingleSnapshot()
    {
        string repoDir = CacheRepo("NoRefsGGUF", "Model-Q8_0.gguf");
        Directory.Delete(Path.Combine(repoDir, "refs"), recursive: true);

        string path = HfCacheResolver.ResolveModelPath(Spec("NoRefsGGUF"), cacheRoot: _root);

        Assert.EndsWith("Model-Q8_0.gguf", path, StringComparison.Ordinal);
    }

    [Fact]
    public void ResolveModelPath_MissingRefs_MultipleSnapshots_RefusesToGuess()
    {
        string repoDir = CacheRepo("AmbiguousGGUF", "Model-Q8_0.gguf");
        Directory.Delete(Path.Combine(repoDir, "refs"), recursive: true);
        Directory.CreateDirectory(Path.Combine(repoDir, "snapshots", new string('a', 40)));

        ArgumentException ex = Assert.Throws<ArgumentException>(() =>
            HfCacheResolver.ResolveModelPath(Spec("AmbiguousGGUF"), cacheRoot: _root));

        Assert.Contains("2 snapshots", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ResolveModelPath_RepoWithoutSnapshot_SaysSo()
    {
        string repoDir = Path.Combine(_root, "models--acme--EmptyGGUF");
        Directory.CreateDirectory(repoDir);

        ArgumentException ex = Assert.Throws<ArgumentException>(() =>
            HfCacheResolver.ResolveModelPath(Spec("EmptyGGUF"), cacheRoot: _root));

        Assert.Contains("no usable snapshot", ex.Message, StringComparison.Ordinal);
    }

    // ---- cache-root discovery ----

    [Fact]
    public void ResolveCacheRoot_FollowsTheDocumentedEnvPrecedence()
    {
        // The same chain huggingface_hub documents; an explicit argument always wins.
        Assert.Equal("/explicit", HfCacheResolver.ResolveCacheRoot("/explicit"));

        var hub = EnvVar("HF_HUB_CACHE", "/a");
        var legacy = EnvVar("HUGGINGFACE_HUB_CACHE", "/b");
        var home = EnvVar("HF_HOME", "/c");
        var xdg = EnvVar("XDG_CACHE_HOME", "/d");
        try
        {
            Assert.Equal("/a", HfCacheResolver.ResolveCacheRoot());
            hub.Value = null;
            Assert.Equal("/b", HfCacheResolver.ResolveCacheRoot());
            legacy.Value = null;
            Assert.Equal(Path.Combine("/c", "hub"), HfCacheResolver.ResolveCacheRoot());
            home.Value = null;
            Assert.Equal(Path.Combine("/d", "huggingface", "hub"), HfCacheResolver.ResolveCacheRoot());
            xdg.Value = null;
            // With nothing set, the home-relative default is used (non-empty here).
            Assert.EndsWith(Path.Combine(".cache", "huggingface", "hub"), HfCacheResolver.ResolveCacheRoot()!,
                StringComparison.Ordinal);
        }
        finally
        {
            hub.Dispose();
            legacy.Dispose();
            home.Dispose();
            xdg.Dispose();
        }
    }

    [Fact]
    public void ResolveModelPath_UsesHfHubCacheEnvironmentVariable()
    {
        CacheRepo("EnvGGUF", "Model-Q8_0.gguf");
        using var env = new EnvScope();
        env.Set("HF_HUB_CACHE", _root);
        // Also clear competing variables so the precedence under test is the one
        // this test exercised.
        using var legacy = new EnvScope();
        legacy.Set("HUGGINGFACE_HUB_CACHE", null);
        legacy.Set("HF_HOME", null);

        string path = HfCacheResolver.ResolveModelPath(Spec("EnvGGUF"));

        Assert.EndsWith("Model-Q8_0.gguf", path, StringComparison.Ordinal);
    }

    // ---- companion discovery ----

    [Fact]
    public void FindCompanionPath_FindsTheProjectorCachedBesideTheModel()
    {
        CacheRepo("VisionGGUF", "Model-Q8_0.gguf", "mmproj-Model-F16.gguf");

        string model = HfCacheResolver.ResolveModelPath(Spec("VisionGGUF"), cacheRoot: _root);
        string? mmproj = HfCacheResolver.FindCompanionPath(model, "mmproj");

        Assert.NotNull(mmproj);
        Assert.EndsWith("mmproj-Model-F16.gguf", mmproj, StringComparison.Ordinal);
    }

    [Fact]
    public void FindCompanionPath_TagPreference_RanksTheAlikeQuantFirst()
    {
        CacheRepo("VisionGGUF", "Model-Q8_0.gguf", "mmproj-Model-F16.gguf", "mmproj-Model-Q8_0.gguf");

        string model = HfCacheResolver.ResolveModelPath(Spec("VisionGGUF"), cacheRoot: _root);
        string? mmproj = HfCacheResolver.FindCompanionPath(model, "mmproj", tag: "Q8_0");

        Assert.NotNull(mmproj);
        Assert.EndsWith("mmproj-Model-Q8_0.gguf", mmproj, StringComparison.Ordinal);
    }

    [Fact]
    public void FindCompanionPath_NothingCached_ReturnsNullInsteadOfThrowing()
    {
        CacheRepo("TextOnlyGGUF", "Model-Q8_0.gguf");

        string model = HfCacheResolver.ResolveModelPath(Spec("TextOnlyGGUF"), cacheRoot: _root);

        Assert.Null(HfCacheResolver.FindCompanionPath(model, "mmproj"));
    }

    [Fact]
    public void ResolveCompanionPath_Spec_SelectsTheProjectorAndHonoursTheTag()
    {
        CacheRepo("VisionGGUF", "mmproj-Model-F16.gguf", "mmproj-Model-Q8_0.gguf");

        string tagged = HfCacheResolver.ResolveCompanionPath(Spec("VisionGGUF:Q8_0"), "mmproj", cacheRoot: _root);
        Assert.EndsWith("mmproj-Model-Q8_0.gguf", tagged, StringComparison.Ordinal);

        string untagged = HfCacheResolver.ResolveCompanionPath(Spec("VisionGGUF"), "mmproj", cacheRoot: _root);
        Assert.EndsWith(".gguf", untagged, StringComparison.Ordinal);
    }

    [Fact]
    public void ResolveCompanionPath_NoProjector_ListsWhatIsCached()
    {
        CacheRepo("TextOnlyGGUF", "Model-Q8_0.gguf");

        ArgumentException ex = Assert.Throws<ArgumentException>(() =>
            HfCacheResolver.ResolveCompanionPath(Spec("TextOnlyGGUF"), "mmproj", cacheRoot: _root));

        Assert.Contains("mmproj", ex.Message, StringComparison.Ordinal);
    }

    private static EnvVarHandle EnvVar(string name, string value)
    {
        var handle = new EnvVarHandle(name);
        handle.Value = value;
        return handle;
    }

    private sealed class EnvVarHandle : IDisposable
    {
        private readonly string _name;
        private readonly string? _original;

        public EnvVarHandle(string name)
        {
            _name = name;
            _original = Environment.GetEnvironmentVariable(name);
        }

        public string? Value
        {
            get => Environment.GetEnvironmentVariable(_name);
            set => Environment.SetEnvironmentVariable(_name, value);
        }

        public void Dispose() => Environment.SetEnvironmentVariable(_name, _original);
    }
}
