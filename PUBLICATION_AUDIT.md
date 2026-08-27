# Publication audit

Audit date: 2026-08-27

## Automated results

- Secret patterns: **PASS** — no API-key assignments, access-token assignments, private-key
  headers, GitHub token forms, or OpenAI-style secret keys detected.
- Email-address scan: **PASS** — no email addresses detected in the publication tree.
- Runtime boundary: **PASS** — no Python import of non-OPM `rlmgraph` modules or `tgram` modules.
- Source boundary: **PASS** — the exported source tree contains only `src/rlmgraph/opm/*.py`.
- GitHub object boundary: **PASS** — no staged publication file exceeds 100,000,000 bytes.
- OPM tests: **PASS** — 96 passed; 15 dependency warnings; 0 failed.
- Public export and closeout verification: **PASS**.

## Byte-bound local-path exception

Three immutable historical artifacts contain `C:/Users/Dwight/...` or the equivalent escaped Windows
path. No credential, email address, or additional user identifier occurs in those paths.

| Artifact | SHA-256 | Why it is retained |
|---|---|---|
| `studies/v1.1.4/workspace/evidence/implementation_validation/OPM_V1_1_4_PRIMARY_EXECUTION_HANDOFF.md` | `9c14ba53688f2c3720a88c01e0a09df87699ae962ae80b8dbb31256dfc6cf23f` | Bound final-report input |
| `studies/v1.1.4/workspace/evidence/primary_runs/v1.1.4/post-primary/stage1-artifacts-v5/run-ddfc05137e09a402/execution.log.jsonl` | `d523ba035b5ef45878209b80c6df215c2b4c08be8e72b58c54cbbe865eda2110` | Bound Stage-1 freeze artifact |
| `studies/v1.1.4/workspace/evidence/primary_runs/v1.1.4/post-primary/stage1-artifacts-v5/run-ddfc05137e09a402/preflight.json` | `b0d750dbe2080bfa12e2ffa9d17bb192adbfe57656dddacf4aa8307c78a6389f` | Bound Stage-1 freeze artifact |

Redacting these paths would invalidate the frozen artifact hashes and the published closeout chain.
They are therefore documented as a narrow historical exception. Repository-owner acceptance of this
exception is required before the repository becomes public.

## External evidence

The Git publication includes 606 manifest-bound snapshot files totaling 6,358,951 bytes. The
external-evidence manifest identifies 444 excluded artifacts totaling approximately 26.5 GiB. These
include selected checkpoints, full training event streams, generated canonical datasets, and
row-level Stage-1 predictions, interventions, and probes.

The external archive remains pending. The repository must not claim that the complete raw evidence
bundle is downloadable until that archive is published and its persistent identifier is recorded.
