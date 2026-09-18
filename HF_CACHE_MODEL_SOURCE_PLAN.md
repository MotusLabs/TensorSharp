# Hugging Face cache as a model source (`-hf org/repo[:quant]`)

## Problem

Every model-path option in TensorSharp today (`--model`, `--mmproj`,
`--draft-model`) is treated as a literal filesystem path —
`ServerOptionsBuilder.ResolveConfiguredModelPath` just does
`Path.GetFullPath(configuredPath)` — and that path is handed straight to
`new GgufFile(path)` by `ModelBase.Create` / `EmbeddingModel.Load`. There is
no concept of a Hugging Face repo id anywhere in the codebase, so a model
already pulled into the local HF cache (via `huggingface-cli download`,
`hf download`, or the `huggingface_hub` Python client) can't be referenced by
name — the user has to find the actual blob/snapshot path on disk first.

llama.cpp solves this with `-hf <user>/<model>[:quant]`
(`common/hf-cache.cpp` + `common/download.cpp`). This plan adapts that
approach to TensorSharp, scoped to **reading an existing local cache only** —
no network calls, no auto-download, no auth-token handling. If a repo/quant
isn't cached, the CLI errors with a clear message instead of fetching it.
Auto-download is a possible future phase, not part of this one.

## Cache layout being read

Standard Hugging Face hub cache layout
(https://huggingface.co/docs/huggingface_hub/guides/manage-cache):

```
<cache_root>/
  models--<org>--<repo>/
    refs/
      main                  # file containing a commit sha
    snapshots/
      <sha>/
        model-00001-of-00005.gguf   -> ../../blobs/<oid>   (symlink)
        model-00002-of-00005.gguf   -> ../../blobs/<oid>
        mmproj-F16.gguf              -> ../../blobs/<oid>
    blobs/
      <oid>                 # actual file content
```

`<cache_root>` resolution order (matches `huggingface_hub` / llama.cpp):
`HF_HUB_CACHE` → `HUGGINGFACE_HUB_CACHE` → `HF_HOME/hub` →
`XDG_CACHE_HOME/huggingface/hub` → `~/.cache/huggingface/hub`
(`%USERPROFILE%` on Windows).

## 1. New resolver — `TensorSharp.Runtime/HuggingFace/HfCacheResolver.cs`

> Implementation note: the plan originally placed this in `TensorSharp.Models`,
> but `SpeculativeCliFlags` — the single shared parse point for `--draft-model`
> used by both hosts — lives in `TensorSharp.Runtime`, and `TensorSharp.Models`
> depends on `TensorSharp.Runtime`, never the reverse. The resolver therefore
> went into `TensorSharp.Runtime/HuggingFace/`, where both `ServerOptionsBuilder`
> and `SpeculativeCliFlags` can reach it.

Pure filesystem + regex; no new dependencies (no HTTP client, no JSON).

- **Cache dir discovery**: env var precedence above.
- **Spec parsing**: `org/repo[:tag]`, split on the last `:`. Validate repo-id
  shape (`^[\w.-]+/[\w.-]+$`, exactly one slash) *before* it feeds path
  construction, to reject path traversal via a crafted `-hf` value.
- **Folder mapping**: `models--{org}--{repo}` (`/` → `--`).
- **Ref resolution**: read `refs/main` (or the first available ref file) to
  get the commit dir under `snapshots/`. Fallback when `refs/` is missing or
  empty: if `snapshots/` has exactly one subdirectory, use it (covers caches
  populated by tools that skip writing refs).
- **File enumeration**: recursively list the snapshot dir; symlinks resolve
  transparently via `File.Exists`/`FileInfo` on both Linux and Windows.
- **Quant matching** (over `*.gguf`, excluding filenames containing
  `mmproj`, `imatrix`, and known draft-sidecar markers):
  - explicit tag → regex `{tag}[.\-]`, case-insensitive;
  - no tag → try `Q4_K_M`, then `Q8_0`, then the first `.gguf` found;
  - for a split file (`name-00001-of-00005.gguf`), only match/return
    **shard 1** — `GgufReader.cs` (`TensorSharp.Runtime/GgufReader.cs:147-190`)
    already auto-discovers and merges the remaining shards from the
    `split.count` GGUF metadata once given shard 1's path, so the resolver
    does not need to reimplement shard grouping.
- **mmproj auto-discovery**: when `-hf` is used and `--mmproj` isn't given
  explicitly, look for a sibling `*mmproj*.gguf` in the same snapshot dir
  (directory-proximity heuristic, simplified version of llama.cpp's
  `find_best_sibling` — no remote-candidate comparison needed here).
- **Not-cached error**: if the repo folder or a matching quant isn't found,
  throw with a clear message listing any `.gguf` files that *are* present
  locally (if the repo is partially cached) and suggesting
  `huggingface-cli download org/repo` / `hf download org/repo`.

## 2. CLI surface

Add to `ServerOptionsBuilder.ParseArgs`
(`TensorSharp.Server/Hosting/ServerOptionsBuilder.cs:987`), alongside the
existing `--model` / `--mmproj` / `--draft-model` handling:

- `-hf` / `--hf-repo <org>/<repo>[:quant]` — mutually exclusive with
  `--model` (error if both given, same style as the existing
  `--embeddings` + `--mmproj` guard at line 69).
- `--hf-file <relative/path.gguf>` — optional, overrides quant matching with
  an exact filename (mirrors llama.cpp's `-hff`).
- Resolution happens once, in `ResolveConfiguredModelPath`: if `-hf` was
  given, call `HfCacheResolver` instead of `Path.GetFullPath`; the result
  feeds `StartupModelPath` exactly as today, so every downstream consumer
  (`ModelBase.Create`, `EmbeddingModel.Load`,
  `ServerOptionsBuilder.ResolvePrefixCacheDirectory`/`ModelCacheKey`, all the
  protocol adapters) needs **zero changes** — they only ever see a resolved
  local path.
- `ResolveConfiguredMmProjPath` gets the same treatment for an `hf:`-prefixed
  (or bare `org/repo[:tag]`-shaped) `--mmproj` value, plus the
  auto-discovery fallback above when `-hf` is used and `--mmproj` is
  omitted.
- `--draft-model` accepts the same spec syntax by routing through the
  resolver too.

## 3. Where it plugs in vs. where it doesn't

`ModelBase.Create`, `EmbeddingModel.Load`, and the architecture factories
stay untouched — they keep taking a plain local path. Resolution is a
CLI-layer concern only, done once in `ServerOptionsBuilder`. Benchmarks
(`AgentTurnBench`, `ChunkParityProbe`, `TensorAgentTtftBench`) that parse
their own `--model` can optionally call the same `HfCacheResolver` for
parity, but that's follow-up polish, not required for the feature to work
end-to-end through the server.

## 4. Tests (`InferenceWeb.Tests`)

- New `HfCacheResolverTests.cs`: build a fake cache tree under a temp dir
  (`models--org--repo/refs/main`, `snapshots/<sha>/...gguf`, a split-shard
  trio, an `mmproj-*.gguf` sibling) and assert:
  - explicit-tag matching,
  - default-quant fallback order (`Q4_K_M` → `Q8_0` → first `.gguf`),
  - split-shard resolution picks index 1,
  - missing-repo / missing-quant error message lists locally cached files,
  - missing-`refs/` single-snapshot fallback,
  - path-traversal-unsafe repo-id values are rejected.
- Extend `ServerOptionsBuilderTests.cs`:
  - `-hf` parses into a resolved path,
  - `-hf` + `--model` together errors,
  - `-hf` + `--mmproj` auto-discovery,
  - `--hf-file` bypasses quant matching.

## 5. Docs

> Implementation note: the operator-facing flag reference lives in `USAGE.md`
> (not `DEVELOPMENT.md`, which has no per-flag section), so the rows went there
> — CLI and server tables, English and `USAGE_zh-cn.md` — plus a
> "Run straight from the Hugging Face cache" section in `MODEL_DOWNLOADS.md`
> (and `MODEL_DOWNLOADS_zh-cn.md`), where the `hf download` recipes already live.

Add `-hf` / `--hf-repo` / `--hf-file` to the CLI section of
`DEVELOPMENT.md` (and `DEVELOPMENT_zh-cn.md`) next to the existing
`--model`/`--mmproj` docs, with the
`unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS` example.

## Deferred (not building now)

Auto-download-on-miss (full llama.cpp `-hf` parity: HTTP client,
ETag/resume, HF token auth, blob/snapshot writing) — a natural Phase 2 if
wanted later. Left out of this pass to avoid adding a network-facing code
path — and its own security review — to this change.
