# Code Context

## Files Retrieved
1. `progress.md` (lines 1-1) - only non-Git file present; contains a prior research-progress note, not project source.
2. `.git/config` (lines 1-5) - confirms this is a local Git repository with default core settings and no remote configured.
3. `.git/info/exclude` (lines 1-6) - only default Git exclude template; no project-specific ignore rules.

## Key Code
No implementation code exists in the working tree.

Observed repository state:
```text
/Volumes/tinyguy/GitHub/brouter
├── .git/
└── progress.md
```

Git state:
```text
## No commits yet on master
?? progress.md
```

Other findings:
- No tracked files (`git ls-files --stage` returned no output).
- No commits yet (`git log` reports the current branch has no commits).
- Current branch: `master`.
- No `origin` or other Git remote is configured.
- No detected package manifests or build files such as `package.json`, `pyproject.toml`, `Cargo.toml`, `go.mod`, `pom.xml`, `build.gradle`, `Makefile`, or `Dockerfile`.

## Architecture
There is no existing application architecture, runtime, or language choice encoded in the repository. The repo is effectively a new empty Git repository plus this progress file.

Given the repository name `brouter`, likely implementation shapes could fit either:
- a small HTTP LLM router/gateway service, or
- a local CLI/library for routing requests to model backends.

Because no source files or manifests exist, the next implementation should be treated as greenfield. A practical plan would be:
1. Choose target runtime and interface first (for example TypeScript/Node for an HTTP gateway, Python/FastAPI for quick service prototyping, Go/Rust for a single static binary).
2. Add baseline project files: README, license if needed, `.gitignore`, formatter/linter config, test framework, and package/build manifest.
3. Define the core domain interfaces before provider-specific code:
   - provider registry
   - routing policy/config schema
   - request/response normalization
   - health checks/fallback behavior
   - observability/logging hooks
4. Implement one minimal route/provider path end-to-end, then add tests and additional providers.
5. Add deployment/dev ergonomics if it is a service: env config, sample config, Dockerfile/compose, and CI.

## Start Here
Start with the repository root (`/Volumes/tinyguy/GitHub/brouter`) because there are no source files yet. The first concrete file to create should be the language/runtime manifest for the chosen stack (`package.json`, `pyproject.toml`, `go.mod`, etc.), followed by a README that states the intended scope of `brouter`.
