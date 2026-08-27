# OPM v1.1.4 post-primary Stage 1 handoff

## Authoritative state

```yaml
lifecycle_state: ARTIFACTS_RECONCILED_AND_FROZEN
authoritative_transition: authorization-v5/OPM_V1_1_4_POST_PRIMARY_TRANSITION.json
transition_sha256: ddfc05137e09a4020560c20ac3e2fe8ac6fcaa8bdf86d4b2947007f0e26f1da6
job_id: 7c9d291371a0e0874ac39eb2
freeze_root: stage1-artifacts-v5/run-ddfc05137e09a402/OPM_V1_1_4_POST_PRIMARY_STAGE1_FREEZE.json
freeze_sha256: 5af3c1243e8cf16c9b5f74dc2ee89412b6e7507fce189f774f6fb835ef65d8ba
merkle_root: 74894ded7e5af418da88cd0f80f40068812ed33091da4d53dad2eaa72e4dbe4e
primary_matrix_sha256: 79107e31d9faee4d8f42685f801b63b503970dace55879de3c087f506338dc82
selected_checkpoints: 20
planned_artifact_groups: 380
frozen_files: 765
sealed_label_access_count: 0
aggregate_test_evaluation_performed: false
claim_decisions_performed: false
```

## Artifact accounting

| Artifact class | Files | Rows |
|---|---:|---:|
| Canonical neural probes | 20 | 40 |
| Label-blind predictions | 80 | 1,032,000 |
| Intervention predictions | 280 | 17,043,520 |
| **Total generated result artifacts** | **380** | **18,075,560** |

Intervention rows comprise 4,902,000 ablation rows, 10,836,000 replacement rows,
1,032,000 adapter-only rows, 81,520 interchange rows, and 192,000 surface-reversal rows.

The reconciliation report records exact 4 x 5 checkpoint coverage, exact artifact identity-set
coverage, valid schemas, finite outputs, matching model pre/post hashes, zero sealed-label fields,
zero aggregate-scoring artifacts, and zero failed jobs represented as complete.

## Verification

```yaml
post_primary_fail_closed_tests: 15 passed
all_opm_tests: 77 passed
ruff: PASS
independent_frozen_file_rehash: 765 checked, 0 mismatches
reconciliation: PASS
```

## Read order

1. `authorization-v5/OPM_V1_1_4_POST_PRIMARY_TRANSITION.json`
2. `authorization-v5/OPM_V1_1_4_SELECTED_CHECKPOINTS.json`
3. `stage1-artifacts-v5/run-ddfc05137e09a402/reconciliation.json`
4. `stage1-artifacts-v5/run-ddfc05137e09a402/OPM_V1_1_4_POST_PRIMARY_STAGE1_FREEZE.json`
5. The frozen `probes/`, `predictions/`, and `interventions/` trees named by the freeze root.

The earlier `authorization`, `authorization-v2`, `authorization-v3`, and `authorization-v4`
directories and their corresponding execution roots are preserved diagnostic attempts. They failed
before a complete Stage 1 freeze and are not authoritative scientific evidence.

## Next authorization boundary

No sealed-test accuracy, bootstrap interval, H1 outcome, mechanism-threshold decision, or scientific
claim was computed in Stage 1. The next operation requires a separate authorization for a distinct
aggregate-only evaluator that consumes the frozen prediction bundle, sealed target bundle, and locked
aggregate specification. That evaluator may not load checkpoints or regenerate predictions. A later,
separate authorization remains required before any claim decision.
