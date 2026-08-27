# Storage profile

Measurements below were taken on 2026-08-27 from the completed local OPM/RLMGraph working tree. The
historical local totals are operational observations, not cryptographically bound publication totals.

| Scope | Bytes | Decimal size | Binary size |
|---|---:|---:|---:|
| Manifest-bound frozen Git snapshot | 6,358,951 | 6.36 MB | 6.06 MiB |
| External manifest's selected excluded artifacts | 28,469,738,283 | 28.47 GB | 26.51 GiB |
| Completed local training/evidence tree | 297,982,329,531 | 297.98 GB | 277.52 GiB |
| Complete local RLMGraph development tree | 306,877,605,685 | 306.88 GB | 285.80 GiB |

The completed evidence tree is therefore not over 500 GB. Its largest v1.1.4 components were pilots
(146.64 GB), primary runs (122.20 GB), and post-primary processing (28.56 GB).

These measurements do not guarantee a future rerun's peak usage. Agents planning a full reproduction
should reserve at least 400 GB for one retained artifact set and environment, with 500 GB providing
safer headroom for temporary files, additional checkpoints, or retries.
