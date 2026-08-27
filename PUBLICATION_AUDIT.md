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
- Clean-clone verification: **PASS** — commit `cb0cb29` was cloned to a new local directory;
  the export verifier reproduced the authoritative hashes and all 96 OPM tests passed.
- Windows checkout note: the preserved evidence tree has deep paths. The successful clean clone used
  a short destination and Git's `core.longpaths=true` setting.
- License: **Apache-2.0** — selected by the repository owner and recorded in `LICENSE` and
  `CITATION.cff`.
- Agent onboarding: **PASS** — doctor, setup dry-run, closed-result verification, and the agent-facing
  96-test command completed successfully.
- GitHub clean-checkout CI: **PASS** — both `Verify frozen OPM study` and `OPM source tests` completed
  successfully on Ubuntu for commit `466a961`.

## Byte-bound local-path exception

Three immutable historical artifacts contain `C:/Users/Dwight/...` or the equivalent escaped Windows
path. No credential, email address, or additional user identifier occurs in those paths.

| Artifact | SHA-256 | Why it is retained |
|---|---|---|
| `studies/v1.1.4/workspace/evidence/implementation_validation/OPM_V1_1_4_PRIMARY_EXECUTION_HANDOFF.md` | `9c14ba53688f2c3720a88c01e0a09df87699ae962ae80b8dbb31256dfc6cf23f` | Bound final-report input |
| `studies/v1.1.4/workspace/evidence/primary_runs/v1.1.4/post-primary/stage1-artifacts-v5/run-ddfc05137e09a402/execution.log.jsonl` | `d523ba035b5ef45878209b80c6df215c2b4c08be8e72b58c54cbbe865eda2110` | Bound Stage-1 freeze artifact |
| `studies/v1.1.4/workspace/evidence/primary_runs/v1.1.4/post-primary/stage1-artifacts-v5/run-ddfc05137e09a402/preflight.json` | `b0d750dbe2080bfa12e2ffa9d17bb192adbfe57656dddacf4aa8307c78a6389f` | Bound Stage-1 freeze artifact |

Redacting these paths would invalidate the frozen artifact hashes and the published closeout chain.
They are therefore documented as a narrow historical exception. The repository owner explicitly
accepted all three disclosures on 2026-08-27.

## External evidence

The Git publication includes 606 manifest-bound snapshot files totaling 6,358,951 bytes. The
external-evidence manifest identifies 444 excluded artifacts totaling approximately 26.5 GiB. These
include selected checkpoints, full training event streams, generated canonical datasets, and
row-level Stage-1 predictions, interventions, and probes.

The external archive remains pending. The repository must not claim that the complete raw evidence
bundle is downloadable until that archive is published and its persistent identifier is recorded.

## Storage observation

The completed local development tree measured 306,877,605,685 bytes, including 297,982,329,531 bytes
under `evidence`. The largest study components were pilots (146.64 GB), primary runs (122.20 GB), and
post-primary processing (28.56 GB). These local measurements are operational observations rather than
manifest-bound publication identities; `docs/storage.md` keeps that distinction explicit.
