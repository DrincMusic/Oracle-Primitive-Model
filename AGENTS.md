# Agent operating contract

This repository is the standalone publication package for the Oracle Primitive Model (OPM). It is
not the RLMGraph application. Treat the closed OPM v1.1.4 result and any future reproduction work as
separate tasks.

## Start here

1. Run `python tools/opm_agent.py doctor`.
2. Read `OPM_EXPERIMENT.json` and `docs/agent-setup.md`.
3. Choose exactly one profile:
   - Closed-result verification: `python tools/opm_agent.py verify`.
   - Development setup and source tests: `python tools/opm_agent.py setup`.
   - Full computational reproduction: stop at the planning stage and follow the blocker described
     below.

## Immutable boundary

`studies/v1.1.4/workspace/` is a byte-preserving historical snapshot. Do not edit, rename, format,
or generate files inside it. The public export manifest binds every non-cache file in that tree.
Place new tools, documentation, and future experiments outside the snapshot.

## Full-reproduction blocker

The Git repository is sufficient to verify the closed result and run the source tests. It is not yet
a one-command full retraining bundle. Large row-level evidence and selected checkpoints are excluded
from Git and recorded in `OPM_EXTERNAL_EVIDENCE_MANIFEST.json`; the persistent external archive is
pending. The historical v1.1.4 workspace is closed and its primary-launch template is disabled.

Do not claim a full reproduction, reset the historical matrices, or launch training from the frozen
workspace. A fresh reproduction workspace and runbook must be published separately and must write to
new output directories.

## Expected verification identity

- Closeout freeze SHA-256: `f9ee482d973fca5522940bf35a168333f6b12864e614d185b29201a709ed165f`
- Closeout Merkle root: `126f7a88f3ad31154cb552d6bd829109a2b5f1548daf1438930bf4b3543618b5`
- Frozen source-test baseline: 96 passing tests

Do not push, publish a release, change repository visibility, or upload evidence unless the repository
owner explicitly authorizes that action.
