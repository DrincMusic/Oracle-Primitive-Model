# Oracle Primitive Model

Oracle Primitive Model (OPM) is a shared-primitive architecture for systematic recombination
generalization. This repository preserves the complete verifiable OPM v1.1.4 study snapshot and the
source, tests, authorization records, and evidence summaries needed to inspect the closed result.

## Result

`OPM_SHARED` achieved 99.7811% sealed recombination accuracy, compared with 57.1611% for the
principal `DOMAIN_GENERALIST` control. H1 was supported with a difference of 0.426200 and a 95%
bootstrap interval of [0.301938, 0.492272]. Both frozen interpolation non-inferiority checks passed.

The complete preregistered composite mechanism was **not supported**. The supported behavioral
result and the unsupported complete mechanism claim are deliberately reported separately. See the
[frozen final report](studies/v1.1.4/workspace/evidence/primary_runs/v1.1.4/study-closeout/report-artifacts-v1/run-4a87ff153338c18e/OPM_V1_1_4_FINAL_REPORT.md)
for the full result, limitations, and claim-by-claim decisions.

## Verify the closed study

The repository-wide verifier checks every copied snapshot file against
`OPM_PUBLIC_EXPORT_MANIFEST.json`, rejects unmanifested snapshot files and oversized Git objects,
then runs the original read-only closeout verifier:

```powershell
python tools/verify_public_export.py --repository .
```

The authoritative identities are:

- Closeout freeze SHA-256: `f9ee482d973fca5522940bf35a168333f6b12864e614d185b29201a709ed165f`
- Closeout Merkle root: `126f7a88f3ad31154cb552d6bd829109a2b5f1548daf1438930bf4b3543618b5`
- Final report SHA-256: `1ba064803da7fb9a6ba4434b9ae056e3bec157195c63ee9620b5687f92e5517d`

## Agent-ready setup

The repository includes an agent operating contract, a machine-readable experiment entry point, and a
cross-platform setup tool. An agent should begin with:

```text
python tools/opm_agent.py doctor
```

Use `python tools/opm_agent.py verify` for the closed result or `python tools/opm_agent.py setup` to
create a virtual environment, install dependencies, verify the snapshot, and run the source tests.
See [`AGENTS.md`](AGENTS.md), [`OPM_EXPERIMENT.json`](OPM_EXPERIMENT.json), and the
[agent setup guide](docs/agent-setup.md).

Full computational retraining is a separate profile and is not yet one-command ready. The external
archive and a fresh-workspace reproduction runbook remain pending; agents must not launch training
inside the immutable historical snapshot.

## Repository layout

```text
studies/v1.1.4/workspace/  Byte-preserving historical study workspace
tools/                     Export and verification tooling
docs/                      Architecture and reproducibility guides
AGENTS.md                   Agent operating contract
OPM_EXPERIMENT.json         Machine-readable task and readiness entry point
OPM_PUBLIC_EXPORT_MANIFEST.json
OPM_EXTERNAL_EVIDENCE_MANIFEST.json
```

The snapshot retains the historical `rlmgraph.opm` namespace because renaming it would change the
source bytes used by the closed study. A future standalone package may use a new namespace without
altering v1.1.4.

## Evidence boundary

Git contains code, tests, study documents, authorization records, frozen reports, aggregate metrics,
sealed targets, training summaries, and cryptographic manifests. Checkpoints, full training event
streams, generated canonical datasets, and row-level Stage-1 evidence are listed by SHA-256 in
`OPM_EXTERNAL_EVIDENCE_MANIFEST.json`; their external archival release is pending.

See the [storage profile](docs/storage.md) for the Git footprint, external inventory, and observed
historical training footprint.

## License

This repository is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE).
