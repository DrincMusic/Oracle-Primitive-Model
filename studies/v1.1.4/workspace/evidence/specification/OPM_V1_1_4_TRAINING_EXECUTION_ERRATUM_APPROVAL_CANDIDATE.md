---
artifact: OPM v1.1.4 Training Execution Erratum
version: 1.1.4
status: PENDING_OWNER_CONFIRMATION
base_amendment_version: 1.1.3
lifecycle: PRIMARY_RUNS
primary_training_started: false
---

# OPM v1.1.4 Training Execution Erratum Candidate

## 1. Scope

This candidate resolves two execution ambiguities discovered after Stage 1 authorization and before
any canonical optimizer step: exact balancing with batch size 256 across three domains, and the
duration/selection rule for the frozen six-combination tuning budget. It does not change datasets,
models, losses, metrics, test access, or claim thresholds.

## 2. Balanced sampling unit

TRN-002 balance is exact over a deterministic three-batch macrocycle of 768 examples. Each macrocycle
contains 384 examples per label, 256 per domain, 192 per operation, and 384 each for one-step and
two-step procedures. Within each label, quotas over supported `(domain, operation)` cells are:

| Domain | LOOKUP | REVERSE | CHAIN | LIFT | Total |
|---|---:|---:|---:|---:|---:|
| SET | 32 | 0 | 48 | 48 | 128 |
| SCENE | 32 | 48 | 48 | 0 | 128 |
| PROGRAM | 32 | 48 | 0 | 48 | 128 |

Let domains be ordered `(SET, SCENE, PROGRAM)`. For run seed `s`, define rotation
`r = derive_uint64("opm-v1.1.4", "batch-partition", s) mod 3`. In macrocycle batch `b in {0,1,2}`:

- every supported non-`LOOKUP` `(domain, operation, label)` stratum contributes exactly 16 examples;
- `LOOKUP` counts across domains for label 0 are rotation `r+b` of `(11,11,10)`;
- `LOOKUP` counts across domains for label 1 are the same rotation of `(11,10,11)`.

Thus every batch contains exactly 128 examples per label, 64 per operation, and 128 examples each for
one-step and two-step procedures. Its domain counts are a seed-rotated permutation of `(86,85,85)`.
Every three-batch macrocycle is exactly balanced at 256 examples per domain and has the Section 2
stratum totals.

For each stratum, examples are ordered by SHA-256 of the UTF-8 string
`opm-v1.1.4/sample/{s}/{macrocycle_index}/{stratum}/{reuse_cycle}/{example_id}`. Selection consumes
the lowest unused ranks, advancing `reuse_cycle` and reranking only when that stratum's finite pool is
exhausted. This defines every reused example without replacement within a reuse cycle.

## 3. Terminal tail

A 50,000-step run executes 16,666 complete macrocycles (49,998 batches), constructs macrocycle 16,666
by the same rule, and consumes its batches `b=0` and `b=1` as the terminal tail. Batch rotation is
fixed by the run seed and is not chosen from observed results.

Across the complete run:

- label, operation, and one-step/two-step totals remain exactly equal;
- domain totals are a seed-rotated permutation whose maximum minus minimum is exactly 1;
- every supported non-`LOOKUP` `(domain, operation)` total is identical to its same-operation peer;
- the three `LOOKUP` domain totals are a seed-rotated permutation with maximum minus minimum 1;
- for label-specific non-`LOOKUP` strata, same-operation peer totals are identical;
- for label-specific `LOOKUP` domain strata, maximum minus minimum is at most 1.

These are the complete-run balance invariants. Any other terminal examples or larger difference is a
conformance failure. No supported stratum may substitute for a held-out domain-operation cell.

## 4. Tuning execution

Pilot seed `1101` runs all six declared `(learning_rate, dropout)` combinations for every four model
conditions. Each pilot run uses the canonical 50,000-step duration, validation/checkpoint cadence,
sampler, loss, and checkpoint-selection rule. This is 24 pilot runs with equal budget.

For each combination, calculate the arithmetic mean of the four conditions' selected-checkpoint macro
validation accuracies. Let `M` be the highest arithmetic mean across the six combinations. The tie set
contains every combination whose mean satisfies `M - mean <= 0.001`. From that set, select the lower
learning rate, then the lower dropout. The selected single combination is frozen for all 20 five-seed
primary runs. Pilot runs are not primary-result replicates and cannot enter claim statistics.

## 5. Authorization boundary

Approval authorizes implementation and execution of this sampler and tuning rule under the existing
Stage 1 permissions. Sealed-label access, aggregate test evaluation, and claim decisions remain
prohibited. Failure may trigger checkpoint resume for the same run identity, but not seed replacement,
hyperparameter substitution, shortened duration, or silent restart under a new identity.

Before the first pilot optimizer step, sampler conformance tests must:

1. verify complete 50,000-step marginal counts for every pilot and primary seed;
2. verify that no held-out domain-operation cell is sampled;
3. verify no within-cycle replacement in every finite stratum pool;
4. verify deterministic replay from identical inputs; and
5. verify checkpoint resume produces the exact same next batch and reuse-cycle state, including that
   the terminal tail is exactly batches `b=0` and `b=1` of macrocycle 16,666.

Any failed sampler conformance test blocks all pilot and primary optimizer steps.

Before approval, machine-readable state must remain:

```yaml
execution_status: BLOCKED_PENDING_V1_1_4_APPROVAL
canonical_training_authorized: false
```

An approved v1.1.4 transition changes only these fields to:

```yaml
execution_status: AUTHORIZED
canonical_training_authorized: true
```

## Approval record

```yaml
approval_status: PENDING_OWNER_CONFIRMATION
approved_by: null
approved_candidate_sha256: null
```
