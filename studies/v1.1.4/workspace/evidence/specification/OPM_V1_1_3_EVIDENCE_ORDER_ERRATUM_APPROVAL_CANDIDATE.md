---
document_type: OPM_IMPLEMENTATION_SPEC_ERRATUM
erratum_version: 1.1.3-candidate
base_erratum_version: 1.1.2
charter_version: 1.2.0
lifecycle_state: ERRATUM_REVIEW
approval_status: PENDING_OWNER_CONFIRMATION
scientific_core_status: BLOCKED_PENDING_ERRATUM_APPROVAL
primary_runs_authorized: false
---

# OPM v1.1.3 evidence-position schedule erratum candidate

## 1. Scope and preserved failure

This erratum addresses the v1.1.2 ORC-005 evidence-position failure. V1.1.2 remains immutable and
gate-failed: its evidence-position probe scored 0.5116, its Wilson interval excluded 0.50, and its
exact-binomial p-value `0.00276` crossed the first Holm boundary. No v1.1.2 seed, artifact, threshold,
or result is replaced.

The correction changes only placement of selected evidence facts among already selected distractors.
It changes no latent world, fact content, query, label, procedure, model, loss, statistical test,
threshold, or primary-run boundary.

The immutable approved base is v1.1.2: candidate normative-prefix SHA-256
`1c1df5c46b43985d71d8b7a52f57a47778ef8b21d1df5c5fbcd154de97ff05f9`, candidate whole-file
SHA-256 `29d4d5553cffed9239c1f0a4b679e014f999f2ecae85470b7c6b2139585b5e2c`, and effective
normative-prefix SHA-256 `6833de9ca280856887be7963be35f5292f8ddfb189f95393ce2451314d25b20b`.
Its exact candidate and approval record are archived under `evidence/specification/`.

## 2. Diagnostic basis

Train/validation positive-rate correlations by evidence position were 0.767 for step 1 and 0.868 for
step 2. Equal-world validation accuracy was 0.5119 with a 95% world-cluster bootstrap interval of
`[0.5032, 0.5201]`. `CHAIN` reached 0.5441. The signal is therefore treated as a recurring
construction coupling rather than an isolated validation fluctuation. These post-gate diagnostics do
not alter the failed result.

## 3. Fresh identity

Corrected train and validation data use fresh namespaces:

```text
opm-v1.1.3/<split>/world/<index>
```

No v1.1.2 world, example, fitted probe, manifest, or prediction may be reused as v1.1.3 canonical
evidence. Test-split generation remains prohibited.

## 4. Counterbalanced evidence-position construction

Allocation and v1.1.2 candidate filtering occur first. Within every
`(split, domain, operation, label, query_relation)` stratum, examples retain the canonical rank `j`
already used by the binding schedule. Positive and negative strata restart the same schedule at
`j = 0`.

Let `F` be the approved fact count: 8 for train and validation. For a one-step example:

```text
evidence_position_1(j) = j mod F
evidence_position_2(j) = PAD
```

For a two-step example:

```text
evidence_position_1(j) = j mod F
evidence_position_2(j) =
  (evidence_position_1(j) + 1 + (floor(j / F) mod (F - 1))) mod F
```

The second position uses offsets `1..F-1` and therefore cannot collide with the first. Place the
step-ordered evidence facts into those target slots. Rank distractors using the existing
`fact-order/<example-id>/<canonical-fact>/<occurrence-index>` hash and place them, in rank order, into
the remaining slots in ascending slot order.

Label is used only to define matched allocation strata. It is not passed to either position formula,
the distractor-order hash, model inputs, probe features, or evaluation.

Because positive and negative relation strata have equal sizes and restart the same schedule, their
per-step evidence-position histograms are exactly identical. Joint step-position patterns remain
deterministic and balanced; evidence identity and endpoint matching remain informative.

## 5. Required conformance tests

Implementation validation must verify:

1. all one-step positions are in `0..F-1` and step 2 is PAD;
2. all two-step positions are in `0..F-1` and distinct;
3. the schedule is deterministic for at least `j=0..7999`;
4. positive and negative per-step and joint-position histograms match exactly within every declared
   train and validation stratum;
5. evidence facts occupy their scheduled slots and distractors fill every remaining slot exactly once;
6. symbolic labels, relation/corruption quotas, binding histograms, and endpoint diagnostics remain
   conformant;
7. there is zero world/example overlap with v1.0, v1.1.2, or diagnostic namespaces;
8. unchanged ORC-005 is run once on fresh v1.1.3 train and validation artifacts.

Any failure blocks protocol freeze. It may not trigger seed search, automatic regeneration, threshold
changes, or another run presented as confirmatory.

## 6. Authorization boundary

Approval authorizes implementation validation and fresh v1.1.3 train/validation construction only.
It does not authorize test-split generation, test-label access, protocol freeze, or primary model
training. A fresh scientific-owner decision remains required after all conformance evidence and the
one-shot ORC-005 result are reviewed.

## Approval record

```yaml
approval_status: PENDING_OWNER_CONFIRMATION
approved_by: null
approved_normative_prefix_sha256: null
approval_statement: null
primary_runs_authorized: false
```

The normative-prefix digest is SHA-256 over the exact UTF-8 bytes from the beginning of this file up
to, but excluding, the `## Approval record` heading.
