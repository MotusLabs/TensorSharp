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
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace TensorSharp.Runtime.HuggingFace
{
    /// <summary>
    /// Resolves a Hugging Face model spec of the form <c>&lt;org&gt;/&lt;repo&gt;[:&lt;quant&gt;]</c>
    /// (e.g. <c>unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS</c>) against a model already present
    /// in the LOCAL Hugging Face hub cache, and returns the file's absolute path. This is
    /// read-only: a repo or quant that is not cached produces a descriptive error that
    /// tells the operator how to download it; nothing is ever fetched from the network.
    ///
    /// The cache layout is the standard hub layout documented at
    /// https://huggingface.co/docs/huggingface_hub/guides/manage-cache —
    /// <c>&lt;root&gt;/models--&lt;org&gt;--&lt;repo&gt;/refs/main</c> names a commit whose
    /// <c>snapshots/&lt;commit&gt;</c> directory holds one symlink per file, pointing into
    /// <c>blobs/</c>. The cache root follows the same environment-variable precedence the
    /// official <c>huggingface_hub</c> client uses, so anything that client downloaded is
    /// found here without extra configuration.
    ///
    /// Split GGUFs (<c>NAME-00001-of-00005.gguf</c>) resolve to shard 1 only; the GGUF
    /// reader discovers the remaining shards itself from the header's split metadata, so
    /// this class never has to group them.
    /// </summary>
    public static class HfCacheResolver
    {
        /// <summary>
        /// Parse a spec without touching the filesystem. The quant tag is optional and
        /// separated by the LAST colon, so a tag may itself contain dots and underscores
        /// but never a slash. Anything else (no slash, two slashes, a Windows drive
        /// letter, an empty tag) is not a spec — callers treat the value as a plain path.
        /// </summary>
        public static bool TryParseSpec(string? value, out string repoId, out string? tag)
        {
            repoId = string.Empty;
            tag = null;
            if (string.IsNullOrWhiteSpace(value))
                return false;

            Match m = SpecRegex.Match(value.Trim());
            if (!m.Success)
                return false;

            repoId = m.Groups["org"].Value + "/" + m.Groups["repo"].Value;
            tag = m.Groups["tag"].Success ? m.Groups["tag"].Value : null;
            return true;
        }

        /// <summary>True when <paramref name="value"/> has the shape of an
        /// <c>&lt;org&gt;/&lt;repo&gt;[:&lt;quant&gt;]</c> spec.</summary>
        public static bool LooksLikeHfSpec(string? value)
            => TryParseSpec(value, out _, out _);

        /// <summary>
        /// For flags that accept EITHER a local path or an HF spec (e.g.
        /// <c>--draft-model</c>): a value that names an existing file is returned
        /// unchanged, a plain path is returned unchanged for the caller's own
        /// existence check, and a spec is resolved through the cache — where draft
        /// files (<c>mtp-</c>, <c>dspark-</c>, …) are eligible picks, because a
        /// standalone drafter is exactly what this flag names.
        /// </summary>
        public static string ResolvePathOrSpec(string value, string? cacheRoot = null)
        {
            if (string.IsNullOrWhiteSpace(value) || !TryParseSpec(value, out _, out _))
                return value;
            if (File.Exists(value))
                return value;
            return ResolveModelPath(value, cacheRoot: cacheRoot, includeDraftFileNames: true);
        }

        /// <summary>
        /// Resolve <paramref name="spec"/> to the absolute path of the cached GGUF it
        /// names. With an explicit quant tag, only files carrying that tag match; without
        /// one, the first file quantized <c>Q4_K_M</c>, then <c>Q8_0</c>, then any model
        /// GGUF at all is used (llama.cpp's fallback order).
        /// </summary>
        /// <param name="explicitFile">Exact repo-relative file name (forward slashes,
        /// e.g. <c>sub/dir/model-Q8_0.gguf</c>) that overrides quant matching — the
        /// <c>--hf-file</c> spelling.</param>
        /// <param name="includeDraftFileNames">False (the default) skips sidecar GGUFs
        /// (<c>mmproj</c>, <c>imatrix</c>, and the draft markers) so a repo that ships a
        /// projector or a drafter next to the trunk still resolves to the trunk; true
        /// makes the draft markers eligible, for callers resolving a DEDICATED draft.</param>
        /// <exception cref="ArgumentException">The spec is malformed, or the repo/quant
        /// is not present in the local cache. The message always says what IS cached and
        /// how to download what is missing.</exception>
        public static string ResolveModelPath(string spec, string? explicitFile = null,
            string? cacheRoot = null, bool includeDraftFileNames = false)
        {
            if (!TryParseSpec(spec, out string repoId, out string? tag))
                throw new ArgumentException(
                    $"'{spec}' is not a Hugging Face model spec of the form <org>/<repo>[:<quant>] " +
                    "(example: unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS).");

            string repoDirectory = GetRepoDirectoryOrThrow(repoId, cacheRoot, out string? root);
            string snapshotDirectory = ResolveSnapshotDirectory(repoDirectory, repoId);
            List<CachedFile> files = EnumerateSnapshotFiles(snapshotDirectory);

            if (!string.IsNullOrWhiteSpace(explicitFile))
            {
                string wanted = explicitFile.Replace('\\', '/');
                CachedFile? hit = files
                    .Where(f => string.Equals(f.RelativePath, wanted, StringComparison.Ordinal))
                    .Select(f => (CachedFile?)f)
                    .FirstOrDefault();
                if (hit is null)
                    throw FilesNotFound(repoId, snapshotDirectory, files, $"file '{explicitFile}'");
                return Path.GetFullPath(hit.Value.AbsolutePath);
            }

            List<string> quantOrder = tag is null
                ? new List<string> { "Q4_K_M", "Q8_0" }
                : new List<string> { tag };

            foreach (string quant in quantOrder)
            {
                Regex pattern = new(Regex.Escape(quant) + "[.-]", RegexOptions.IgnoreCase | RegexOptions.Compiled);
                // FirstOrDefault over the struct sequence would yield default(CachedFile) —
                // never null — so the no-match case must be projected to CachedFile? first.
                CachedFile? pick = files
                    .Where(f => IsModelFile(f.RelativePath, includeDraftFileNames)
                                && pattern.IsMatch(f.RelativePath)
                                && (!IsSplitShard(f.RelativePath, out int index, out _) || index == 1))
                    .OrderBy(f => f.RelativePath, StringComparer.Ordinal)
                    .Select(f => (CachedFile?)f)
                    .FirstOrDefault();
                if (pick is not null)
                    return Path.GetFullPath(pick.Value.AbsolutePath);
            }

            // Without a tag, any model GGUF at all is acceptable; with one, only the
            // named quant will do — silently serving a different precision would be worse
            // than failing.
            if (tag is null)
            {
                CachedFile? any = files
                    .Where(f => IsModelFile(f.RelativePath, includeDraftFileNames)
                                && (!IsSplitShard(f.RelativePath, out int index, out _) || index == 1))
                    .OrderBy(f => f.RelativePath, StringComparer.Ordinal)
                    .Select(f => (CachedFile?)f)
                    .FirstOrDefault();
                if (any is not null)
                    return Path.GetFullPath(any.Value.AbsolutePath);
            }

            string asked = tag is null ? "any model GGUF" : $"quant '{tag}'";
            throw FilesNotFound(repoId, snapshotDirectory, files, asked, root);
        }

        /// <summary>
        /// Resolve a companion-file spec (e.g. <c>--mmproj org/vision-GGUF:F16</c>) to a
        /// GGUF in the cached snapshot whose file name contains
        /// <paramref name="keyword"/> ("mmproj" today). The quant tag, when given,
        /// selects between several companions.
        /// </summary>
        public static string ResolveCompanionPath(string spec, string keyword, string? cacheRoot = null)
        {
            if (!TryParseSpec(spec, out string repoId, out string? tag))
                throw new ArgumentException(
                    $"'{spec}' is not a Hugging Face model spec of the form <org>/<repo>[:<quant>] " +
                    "(example: unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS).");

            string repoDirectory = GetRepoDirectoryOrThrow(repoId, cacheRoot, out _);
            string snapshotDirectory = ResolveSnapshotDirectory(repoDirectory, repoId);
            List<CachedFile> files = EnumerateSnapshotFiles(snapshotDirectory);

            List<CachedFile> candidates = files
                .Where(f => f.RelativePath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
                            && FileNameOf(f.RelativePath).Contains(keyword, StringComparison.OrdinalIgnoreCase))
                .OrderBy(f => f.RelativePath, StringComparer.Ordinal)
                .ToList();

            if (candidates.Count == 0)
                throw FilesNotFound(repoId, snapshotDirectory, files, $"a '{keyword}' GGUF");

            if (tag is not null)
            {
                Regex pattern = new(Regex.Escape(tag) + "[.-]", RegexOptions.IgnoreCase | RegexOptions.Compiled);
                CachedFile? tagged = candidates.FirstOrDefault(f => pattern.IsMatch(f.RelativePath));
                if (tagged is null)
                    throw new ArgumentException(
                        $"No '{keyword}' GGUF matching quant '{tag}' was found for repo '{repoId}' in the " +
                        $"Hugging Face cache snapshot at '{snapshotDirectory}'. Available '{keyword}' files: " +
                        string.Join(", ", candidates.Select(f => f.RelativePath)));
                return Path.GetFullPath(tagged.Value.AbsolutePath);
            }

            return Path.GetFullPath(candidates[0].AbsolutePath);
        }

        /// <summary>
        /// Find a companion GGUF (projector, drafter sidecar) sitting NEXT TO an already
        /// resolved model inside its cache snapshot — the auto-discovery behind starting
        /// the server with <c>-hf</c> and no explicit <c>--mmproj</c>. Returns null when
        /// nothing matches; discovery must never be the reason a run fails. Ranking
        /// mirrors llama.cpp's <c>find_best_sibling</c>: prefer an exact quant-tag match,
        /// then the quantization closest to the trunk's.
        /// </summary>
        /// <param name="modelLocalPath">Absolute path of the already-resolved model GGUF.</param>
        /// <param name="keyword">File-name marker to search for, e.g. "mmproj".</param>
        /// <param name="tag">The model spec's quant tag, if one was given, used to
        /// prefer a companion quantized alike.</param>
        public static string? FindCompanionPath(string modelLocalPath, string keyword, string? tag = null)
        {
            if (string.IsNullOrWhiteSpace(modelLocalPath) || !File.Exists(modelLocalPath))
                return null;

            string? snapshotDirectory = FindSnapshotDirectory(modelLocalPath);
            if (snapshotDirectory is null)
                return null;

            string modelRelative = Path.GetRelativePath(snapshotDirectory, modelLocalPath);
            string? modelRelativeDirectory = Path.GetDirectoryName(modelRelative);

            Regex? tagPattern = tag is null
                ? null
                : new Regex(Regex.Escape(tag) + "[.-]", RegexOptions.IgnoreCase | RegexOptions.Compiled);
            int modelBits = ExtractQuantBits(modelLocalPath);

            return Directory
                .EnumerateFiles(snapshotDirectory, "*.gguf", SearchOption.AllDirectories)
                .Where(path => File.Exists(path))
                .Select(path => new
                {
                    Absolute = path,
                    Relative = Path.GetRelativePath(snapshotDirectory, path),
                    Name = Path.GetFileName(path),
                })
                .Where(f => f.Name.Contains(keyword, StringComparison.OrdinalIgnoreCase)
                            && Path.GetDirectoryName(f.Relative) == modelRelativeDirectory)
                .OrderBy(f => tagPattern != null && !tagPattern.IsMatch(f.Relative))   // tagged first
                .ThenBy(f => Math.Abs(ExtractQuantBits(f.Absolute) - modelBits))
                .ThenBy(f => f.Relative, StringComparer.Ordinal)
                .Select(f => f.Absolute)
                .FirstOrDefault();
        }

        /// <summary>
        /// The Hugging Face cache root, using the environment-variable precedence the
        /// official client documents: HF_HUB_CACHE, HUGGINGFACE_HUB_CACHE, HF_HOME/hub,
        /// XDG_CACHE_HOME/huggingface/hub, then ~/.cache/huggingface/hub. Null when even
        /// the home directory cannot be located.
        /// </summary>
        public static string? ResolveCacheRoot(string? cacheRoot = null)
        {
            if (!string.IsNullOrWhiteSpace(cacheRoot))
                return cacheRoot;

            string? value;
            if (!string.IsNullOrWhiteSpace(value = Environment.GetEnvironmentVariable("HF_HUB_CACHE")))
                return value;
            if (!string.IsNullOrWhiteSpace(value = Environment.GetEnvironmentVariable("HUGGINGFACE_HUB_CACHE")))
                return value;
            if (!string.IsNullOrWhiteSpace(value = Environment.GetEnvironmentVariable("HF_HOME")))
                return Path.Combine(value, "hub");
            if (!string.IsNullOrWhiteSpace(value = Environment.GetEnvironmentVariable("XDG_CACHE_HOME")))
                return Path.Combine(value, "huggingface", "hub");

            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            return string.IsNullOrWhiteSpace(home)
                ? null
                : Path.Combine(home, ".cache", "huggingface", "hub");
        }

        // ---- internals ----

        private readonly record struct CachedFile(string RelativePath, string AbsolutePath);

        // One slash, no empty/dotted segments, no separator characters inside the
        // segments: a parsed spec can only ever become ONE path segment under the cache
        // root ("<root>/models--org--repo"), so a crafted -hf value cannot climb out of
        // the cache directory. Tags may repeat the same alphabet (dots included) because
        // a tag never reaches the filesystem as a path.
        private static readonly Regex SpecRegex = new(
            @"^(?<org>[A-Za-z0-9_][A-Za-z0-9_.-]*)/(?<repo>[A-Za-z0-9_][A-Za-z0-9_.-]*)(:(?<tag>[A-Za-z0-9_][A-Za-z0-9_.-]*))?$",
            RegexOptions.Compiled | RegexOptions.ExplicitCapture);

        // llama.cpp's gguf-split naming: "<prefix>-00001-of-00005.gguf".
        private static readonly Regex ShardRegex = new(
            @"^(?<prefix>.+)-(?<no>\d{5})-of-(?<count>\d{5})\.gguf$",
            RegexOptions.Compiled | RegexOptions.ExplicitCapture | RegexOptions.IgnoreCase);

        // The quantization tag at the end of a GGUF name: "...-UD-IQ3_XXS" -> "IQ3_XXS".
        private static readonly Regex QuantTailRegex = new(
            @"[-.]([A-Za-z0-9_]+)$",
            RegexOptions.Compiled | RegexOptions.ExplicitCapture);

        private static readonly string[] ProjectorMarkers = { "mmproj", "imatrix" };
        private static readonly string[] DraftMarkers = { "mtp-", "eagle3-", "dflash", "dspark-" };

        private static bool IsModelFile(string relativePath, bool includeDraftFileNames)
        {
            if (!relativePath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
                return false;
            string name = FileNameOf(relativePath);
            if (ProjectorMarkers.Any(marker => name.Contains(marker, StringComparison.OrdinalIgnoreCase)))
                return false;
            if (!includeDraftFileNames && DraftMarkers.Any(marker => name.Contains(marker, StringComparison.OrdinalIgnoreCase)))
                return false;
            return true;
        }

        private static bool IsSplitShard(string relativePath, out int index, out int count)
        {
            Match m = ShardRegex.Match(FileNameOf(relativePath));
            if (m.Success
                && int.TryParse(m.Groups["no"].Value, out index)
                && int.TryParse(m.Groups["count"].Value, out count))
                return true;
            index = 1;
            count = 1;
            return false;
        }

        // Q4_K_M -> 4, UD-IQ3_XXS -> 3, F16 -> 16; 0 when no digits are present. Mirrors
        // llama.cpp's extract_quant_bits closely enough to rank companion quants.
        private static int ExtractQuantBits(string path)
        {
            string stem = Path.GetFileName(path);
            if (stem.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
                stem = stem[..^".gguf".Length];
            Match shard = ShardRegex.Match(stem);
            if (shard.Success)
                stem = shard.Groups["prefix"].Value;

            Match m = QuantTailRegex.Match(stem);
            if (!m.Success)
                return 0;
            string tag = m.Groups[1].Value;
            int pos = tag.IndexOfAny("0123456789".ToCharArray());
            if (pos < 0)
                return 0;

            int end = pos;
            while (end < tag.Length && char.IsAsciiDigit(tag[end]))
                end++;
            return int.TryParse(tag[pos..end], out int bits) ? bits : 0;
        }

        private static string FileNameOf(string relativePath)
        {
            int slash = relativePath.LastIndexOf('/');
            return slash < 0 ? relativePath : relativePath[(slash + 1)..];
        }

        private static string GetRepoDirectoryOrThrow(string repoId, string? cacheRoot, out string? root)
        {
            root = ResolveCacheRoot(cacheRoot);
            if (root is null)
                throw new ArgumentException(
                    $"Cannot locate the Hugging Face cache directory for repo '{repoId}'. " +
                    "Set HF_HUB_CACHE (or HF_HOME) to the cache root, or pass a local GGUF path instead.");

            string repoDirectory = Path.Combine(root, "models--" + repoId.Replace("/", "--"));
            if (Directory.Exists(repoDirectory))
                return repoDirectory;

            string hint = string.Empty;
            try
            {
                List<string> cached = Directory.EnumerateDirectories(root, "models--*")
                    .Select(folder => FolderNameToRepoId(Path.GetFileName(folder)))
                    .Where(id => id is not null)
                    .Select(id => id!)
                    .OrderBy(id => id, StringComparer.Ordinal)
                    .Take(20)
                    .ToList();
                if (cached.Count > 0)
                    hint = " Repos currently in the cache: " + string.Join(", ", cached) + ".";
            }
            catch (IOException)
            {
            }
            catch (UnauthorizedAccessException)
            {
            }

            throw new ArgumentException(
                $"Model repo '{repoId}' was not found in the local Hugging Face cache (root: '{root}').{hint} " +
                $"Download it first, for example: hf download {repoId} (or: huggingface-cli download {repoId}), " +
                $"then pass it as -hf {repoId} again.");
        }

        // "models--org--repo" back to "org/repo"; null for anything else.
        private static string? FolderNameToRepoId(string? folderName)
        {
            const string prefix = "models--";
            if (folderName is null || !folderName.StartsWith(prefix, StringComparison.Ordinal))
                return null;
            string id = folderName[prefix.Length..].Replace("--", "/");
            return LooksLikeHfSpec(id) ? id : null;
        }

        private static string ResolveSnapshotDirectory(string repoDirectory, string repoId)
        {
            string refsDirectory = Path.Combine(repoDirectory, "refs");
            if (Directory.Exists(refsDirectory))
            {
                string? commit = ReadCommitFile(Path.Combine(refsDirectory, "main"))
                    ?? Directory.EnumerateFiles(refsDirectory)
                        .OrderBy(path => path, StringComparer.Ordinal)
                        .Select(ReadCommitFile)
                        .FirstOrDefault(commit => commit is not null);

                if (commit is not null)
                {
                    string snapshot = Path.Combine(repoDirectory, "snapshots", commit);
                    if (Directory.Exists(snapshot))
                        return snapshot;
                }
            }

            // Caches written by tools that skip the refs file: exactly one snapshot is
            // unambiguous, more than one is a guess we refuse to make.
            string snapshotsDirectory = Path.Combine(repoDirectory, "snapshots");
            if (Directory.Exists(snapshotsDirectory))
            {
                string[] snapshots = Directory.GetDirectories(snapshotsDirectory);
                if (snapshots.Length == 1)
                    return snapshots[0];
                if (snapshots.Length > 1)
                    throw new ArgumentException(
                        $"The cached repo '{repoId}' has no refs/ entry and {snapshots.Length} snapshots; " +
                        "pick one by re-downloading (hf download " + repoId + ") or by clearing the stale " +
                        "snapshots from " + snapshotsDirectory + ".");
            }

            throw new ArgumentException(
                $"The cached repo '{repoId}' contains no usable snapshot (looked in '{repoDirectory}'). " +
                "Re-download it: hf download " + repoId + " (or: huggingface-cli download " + repoId + ").");
        }

        private static string? ReadCommitFile(string path)
        {
            try
            {
                if (!File.Exists(path))
                    return null;
                string commit = File.ReadAllText(path).Trim();
                return IsHex(commit, 40) || IsHex(commit, 64) ? commit : null;
            }
            catch (IOException)
            {
                return null;
            }
        }

        private static bool IsHex(string value, int length)
        {
            if (value.Length != length)
                return false;
            foreach (char c in value)
                if (!Uri.IsHexDigit(c))
                    return false;
            return true;
        }

        private static List<CachedFile> EnumerateSnapshotFiles(string snapshotDirectory)
        {
            var files = new List<CachedFile>();
            foreach (string path in Directory.EnumerateFiles(snapshotDirectory, "*", SearchOption.AllDirectories))
            {
                // Snapshot entries are symlinks into blobs/; a dangling one is not a
                // file we can hand to the loader, so it does not exist as far as
                // matching is concerned.
                if (!File.Exists(path))
                    continue;
                files.Add(new CachedFile(Path.GetRelativePath(snapshotDirectory, path).Replace('\\', '/'), path));
            }
            return files;
        }

        // Walk up from the model file to its snapshot directory: the winner is the
        // directory whose parent holds blobs/ and is named snapshots/. Returns null off
        // the cache layout (a plain local path has no snapshot to search).
        private static string? FindSnapshotDirectory(string modelLocalPath)
        {
            string? directory = Path.GetDirectoryName(Path.GetFullPath(modelLocalPath));
            while (!string.IsNullOrEmpty(directory))
            {
                string? parent = Path.GetDirectoryName(directory);
                if (parent is not null
                    && string.Equals(Path.GetFileName(parent), "snapshots", StringComparison.Ordinal)
                    && Directory.Exists(Path.Combine(Path.GetDirectoryName(parent)!, "blobs")))
                    return directory;
                directory = parent;
            }
            return null;
        }

        private static ArgumentException FilesNotFound(string repoId, string snapshotDirectory,
            List<CachedFile> files, string asked, string? root = null)
        {
            string available = string.Join(", ", files
                .Where(f => f.RelativePath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
                .Select(f => f.RelativePath)
                .OrderBy(p => p, StringComparer.Ordinal));

            string listing = available.Length > 0
                ? $" GGUF files in the cached snapshot: {available}."
                : $" The cached snapshot at '{snapshotDirectory}' contains no GGUF files.";
            string download = root is null
                ? string.Empty
                : $" Download the quant you need (hf download {repoId}) and pass it as -hf {repoId} again.";

            return new ArgumentException(
                $"No {asked} was found for repo '{repoId}' in the local Hugging Face cache.{listing}{download}");
        }
    }
}
