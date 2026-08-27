# OPM v1.1.4 sealed aggregate evaluation handoff

## Authoritative state

```yaml
lifecycle_state: SEALED_LABEL_AGGREGATE_EVALUATION_COMPLETE_AND_FROZEN
authorization: authorization-v4/OPM_V1_1_4_SEALED_AGGREGATE_AUTHORIZATION.json
authorization_sha256: c41bb1e61655891ea7f47b4ecdada1cddfdfb4f10bf121dc4a0f12b6fc93ce0c
aggregate_spec_sha256: fac8bbf6d3ac0f10977e3056e358e003620e5518cc32c9bece1e655ee72fec61
aggregate_freeze: aggregate-artifacts-v4/run-c41bb1e61655891e/OPM_V1_1_4_SEALED_AGGREGATE_FREEZE.json
aggregate_freeze_sha256: e9b3c8bd9007ecc52e46ad985065acfff4b1d22d7530bf67d228c964cc1c972a
aggregate_merkle_root: 2d1fa74c1637f621a748ab69f117d9651fed2facae8c88f919419eae8bea1a9b
prediction_rows_joined: 1032000
intervention_rows_joined: 17043520
sealed_target_rows_joined: 51600
bootstrap_replicates: 10000
checkpoint_access_count: 0
model_load_count: 0
prediction_generation_count: 0
claim_threshold_application_count: 0
claim_decision_count: 0
```

The v4 package is the authoritative aggregate-only result. No H1 threshold, mechanism criterion,
support status, or scientific claim was applied in this lifecycle stage.

## Baseline sealed-test accuracy

Values below pool the five declared model seeds.

| Condition | Interpolation | Recombination | Renderer | Structural |
|---|---:|---:|---:|---:|
| `OPM_SHARED` | 0.999347 | 0.997811 | 0.999354 | 0.998979 |
| `DOMAIN_GENERALIST` | 0.997264 | 0.571611 | 0.590729 | 0.890854 |
| `PROC_UNTIED` | 0.997583 | 0.499989 | 0.738417 | 0.873667 |
| `PROC_CLONE` | 0.997861 | 0.500000 | 0.749292 | 0.873854 |

The complete baseline file also contains every per-seed split result, per-domain result,
per-operation result, one-step/two-step result, renderer-variant result, and all ten calibration bins.

## Paired preregistered effects

| Statistic | Point estimate | 95% percentile interval |
|---|---:|---:|
| `Delta_generalist` | 0.426200 | [0.301938, 0.492272] |
| `Delta_untied` | 0.497822 | [0.493535, 0.501668] |

These use 10,000 two-level bootstrap replicates: resample model seeds, then resample whole world IDs
within each selected seed, retain each world's row multiplicities, compute each withheld-cell
accuracy, and macro-average the three cells.

| Seed | OPM_SHARED | DOMAIN_GENERALIST | PROC_UNTIED | Delta_generalist | Delta_untied |
|---:|---:|---:|---:|---:|---:|
| 1101 | 0.999500 | 0.503000 | 0.500000 | 0.496500 | 0.499500 |
| 2202 | 0.991667 | 0.504111 | 0.500000 | 0.487556 | 0.491667 |
| 3303 | 0.999667 | 0.518056 | 0.499944 | 0.481611 | 0.499722 |
| 4404 | 0.999444 | 0.820333 | 0.500000 | 0.179111 | 0.499444 |
| 5505 | 0.998778 | 0.512556 | 0.500000 | 0.486222 | 0.498778 |

## OPM_SHARED intervention summary

| Intervention | Split | Accuracy | Frozen baseline | Change |
|---|---|---:|---:|---:|
| Adapter-only | Interpolation | 0.500000 | 0.999347 | -0.499347 |
| Adapter-only | Recombination | 0.499733 | 0.997811 | -0.498078 |
| Adapter-only | Renderer | 0.500083 | 0.999354 | -0.499271 |
| Adapter-only | Structural | 0.500125 | 0.998979 | -0.498854 |
| Cross-domain interchange | Renderer pairs | 0.346271 | 0.999166 | -0.652895 |
| Surface reversal | Renderer | 0.999354 | 0.999354 | 0.000000 |

All 80 sentinel-ablation summaries have a maximum absolute logit change of `0.0`.
The intervention file contains 268 overall summaries, 1,340 seed-level summaries, and 2,412
component/domain/operation summaries, including the full ablation and replacement matrices for all
four conditions.

## Frozen probe means

| Condition | Evidence step 1 | Evidence step 2 |
|---|---:|---:|
| `OPM_SHARED` | 0.503031 | 0.496344 |
| `DOMAIN_GENERALIST` | 0.503313 | 0.503219 |
| `PROC_UNTIED` | 0.499219 | 0.500656 |
| `PROC_CLONE` | 0.500844 | 0.502844 |

The probe artifact preserves all 40 frozen validation-scale results, including counts, p-values,
and Wilson intervals. No mechanism criterion was applied.

## Verification

```yaml
opm_tests: 85 passed
ruff: PASS
frozen_result_files_rehashed: 8
rehash_mismatches: 0
merkle_recomputed: MATCH
nonfinite_aggregate_values: 0
per_seed_macro_vs_direct_split_max_abs_difference: 1.11e-16
```

## Result files

- `aggregate-artifacts-v4/run-c41bb1e61655891e/baseline-metrics.json`
- `aggregate-artifacts-v4/run-c41bb1e61655891e/primary-effects.json`
- `aggregate-artifacts-v4/run-c41bb1e61655891e/intervention-metrics.json`
- `aggregate-artifacts-v4/run-c41bb1e61655891e/probe-metrics.json`
- `aggregate-artifacts-v4/run-c41bb1e61655891e/execution-summary.json`
- `aggregate-artifacts-v4/run-c41bb1e61655891e/sealed-target-access.audit.jsonl`
- `aggregate-artifacts-v4/run-c41bb1e61655891e/preflight.json`
- `aggregate-artifacts-v4/run-c41bb1e61655891e/environment.json`
- `aggregate-artifacts-v4/run-c41bb1e61655891e/OPM_V1_1_4_SEALED_AGGREGATE_FREEZE.json`

## Diagnostic attempts

Authorization/output versions v1 through v3 are preserved as non-authoritative diagnostics. V1 and
v2 failed closed before aggregate publication. V3 froze a package, but independent review detected
that its bootstrap point estimator equally weighted world-level accuracies despite variable rows per
world; v4 supersedes it with correct within-cell correct/count weighting. No diagnostic package may
be used for claims.

## Next authorization boundary

The next lifecycle operation is `CLAIM_DECISION`. It requires a separate authorization before any
H1 support threshold, interpolation non-inferiority rule, mechanism criterion, or claim wording is
applied to these frozen aggregates.
