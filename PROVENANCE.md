# Provenance

OPM v1.1.4 was originally developed inside the private RLMGraph workspace. Historical Python
namespaces and relative paths are retained inside the frozen study snapshot to preserve byte
identities and reproducibility. TGRAM and non-OPM RLMGraph components are not included in this
repository.

The original OPM files were not represented by the development repository's recorded Git commit.
The study therefore binds its implementation through the declared OPM source-tree digest, file-level
SHA-256 records, immutable authorization packages, artifact freezes, and the final closeout freeze.

The public export was produced into a new Git history. It does not inherit the private RLMGraph Git
history. `OPM_PUBLIC_EXPORT_MANIFEST.json` records every byte-preserved file copied into the study
snapshot. `OPM_EXTERNAL_EVIDENCE_MANIFEST.json` records large artifacts that are intentionally kept
out of ordinary Git history pending an archival release.

Some byte-bound historical handoff records contain original local Windows paths. They are retained
only where the frozen closeout package requires their exact hashes; they are not live configuration
and must not be interpreted as portable paths.
