---
document_type: OPM_IMPLEMENTATION_SPEC_ERRATUM
erratum_version: 1.1.1
base_amendment_version: 1.1.0
charter_version: 1.2.0
lifecycle_state: ERRATUM_APPROVED
approval_status: APPROVED
scientific_core_status: AUTHORIZED_FOR_IMPLEMENTATION_VALIDATION
primary_runs_authorized: false
---

# OPM v1.1.1 object-binding schedule erratum candidate

## 1. Scope

This erratum corrects one arithmetic defect in section 4.3 of
`OPM_V1_1_LEAKAGE_AMENDMENT.md`. It changes no hypothesis, dataset size, allocation quota, model,
loss, statistical test, threshold, seed, or lifecycle gate.

All provisions of the approved v1.0 specification and v1.1 amendment remain normative except the
single expression replaced below.

## 2. Defect

The approved object–object target schedule states:

```text
arg1_target(j) = j mod 32
arg2_target(j) = (arg1_target(j) + 1 + floor(j / 32)) mod 32
```

It also requires distinct query objects to receive distinct targets. Those statements conflict. At
`j = 992`, `floor(j / 32) = 31`, so both targets equal zero. The collision repeats whenever
`floor(j / 32) mod 32 = 31`. Approved train relation strata are large enough to encounter it.

## 3. Corrected normative expression

Replace the second expression with:

```text
arg2_target(j) = (arg1_target(j) + 1 + (floor(j / 32) mod 31)) mod 32
```

The offset is therefore always in `1..31`, so the two targets can never collide. `arg1_target`
continues to cycle over all 32 object IDs. The 31 nonidentity offsets cycle deterministically. Because
positive and negative relation strata restart the identical schedule at `j = 0`, their argument-ID
histograms remain exactly matched.

## 4. Required conformance tests

Implementation validation must exhaustively verify at least `j = 0..7999` that:

1. both targets are in `0..31`;
2. the targets are distinct;
3. the schedule is deterministic;
4. positive and negative strata of equal size receive identical per-position histograms;
5. all v1.1 selected-fact endpoint diagnostic requirements remain satisfied.

## 5. Authorization boundary

Approval authorizes implementation validation of this corrected expression. It does not authorize
protocol freeze, primary model training, or test-label evaluation. The corrected v1.1 binding
implementation is now authorized for implementation validation.

## Approval record

```yaml
approval_status: APPROVED
approved_by: Dwight Robert Keller-Williams
approved_candidate_normative_prefix_sha256: b459e967c020fa20e15f46736903101927ea360955cc3e50a9283d69c9bb88f6
approved_candidate_whole_file_sha256: e15f0a0a1c671ff8714cf95a8202bb092035e49b55794eec44ff6ac46c0c56c7
effective_post_transition_normative_prefix_sha256: a97fd20dc176d71ba4123d1ffaa8c143f3f63cae71fe3492ae92ccd31ccb3f8c
approval_statement: I, Dwight Robert Keller-Williams approve OPM v1.1.1 Object-Binding Schedule Erratum with normative-prefix SHA-256 b459e967c020fa20e15f46736903101927ea360955cc3e50a9283d69c9bb88f6.
primary_runs_authorized: false
```

The normative-prefix digest is SHA-256 over the exact UTF-8 bytes from the beginning of this file up
to, but excluding, the `## Approval record` heading.
