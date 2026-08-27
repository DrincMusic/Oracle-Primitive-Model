---
document_type: OPM_IMPLEMENTATION_SPEC_ERRATUM
erratum_version: 1.1.2-candidate
base_erratum_version: 1.1.1
charter_version: 1.2.0
lifecycle_state: ERRATUM_REVIEW
approval_status: PENDING_OWNER_CONFIRMATION
scientific_core_status: BLOCKED_PENDING_ERRATUM_APPROVAL
primary_runs_authorized: false
---

# OPM v1.1.2 self-endpoint corruption erratum candidate

## 1. Scope

This erratum corrects one endpoint-corruption edge case discovered while validating the approved
v1.1.1 binding invariant. It changes no hypothesis, split size, model, loss, statistical test,
threshold, seed, or lifecycle gate.

### 1.1 Verified base chain

This candidate applies on top of the following immutable, approved artifacts:

```yaml
v1_1_amendment:
  candidate_normative_prefix_sha256: 479fbe9beb4adba342bfe89ab817d9a86ac097aa536365f6fff0c5808937d845
  candidate_whole_file_sha256: 43015a93667f34da4b881617ea64e28985a9a9a121d102f59c19186b193e9e66
  effective_normative_prefix_sha256: c0bf6807463ea15baabfb968ad2816e863b666b8bd3f3bf0ba9911b7af8d5c8d
v1_1_1_erratum:
  candidate_normative_prefix_sha256: b459e967c020fa20e15f46736903101927ea360955cc3e50a9283d69c9bb88f6
  candidate_whole_file_sha256: e15f0a0a1c671ff8714cf95a8202bb092035e49b55794eec44ff6ac46c0c56c7
  effective_normative_prefix_sha256: a97fd20dc176d71ba4123d1ffaa8c143f3f63cae71fe3492ae92ccd31ccb3f8c
```

Exact pre-transition candidates and separate approval records are preserved under
`evidence/specification/`. The label-stratification clarification and selected-fact endpoint
diagnostic requirement are recorded in the approved v1.1 amendment's post-approval implementation
clarifications. V1.1.1 conformance test 5 explicitly inherits those endpoint diagnostics. V1.1.2
changes only endpoint-candidate eligibility and applies without conflict to both artifacts.

## 2. Defect

The v1.0 endpoint-corruption generator requires a type-compatible replacement distinct from the
original second argument, but does not require it to be distinct from the first argument. For
object/object queries it can therefore produce examples such as `BEFORE(x,x)`.

Such a query is a valid symbolic negative, but equality of the two oracle argument IDs becomes a
direct label cue: approved positive `BEFORE`, `SAME`, and `LINK` queries use distinct entities. An
example-scoped bijection must map one latent entity to one ID, so the v1.1 counterbalancer cannot remove
this cue without violating entity-identity consistency.

## 3. Corrected normative rule

For `ENDPOINT` corruption, the replacement for query argument 2 must satisfy all of:

```text
replacement_entity_type == original_argument_2_entity_type
replacement != original_argument_2
replacement != argument_1
```

All other corruption rules remain unchanged. If no replacement satisfies these constraints, that
positive template contributes no `ENDPOINT` candidate. Split allocation continues deterministically
with later candidates or worlds and fails loudly if a declared quota cannot be met.

This rule is applied before candidate ranking and before the v1.1 binding schedule. It is independent
of the model, probe prediction, and observed gate outcome.

## 4. Required conformance tests

Implementation validation must verify that:

1. no generated `ENDPOINT` corruption has identical query arguments;
2. recompute the formal query truth value after corruption using the typed relational semantics and
   assert that every generated `ENDPOINT` corruption candidate evaluates false;
3. approved relation and corruption quotas remain exact;
4. binding identity is preserved when a query legitimately repeats an entity in any future diagnostic;
5. canonical ORC-005 is run once on fresh v1.1 data after all construction checks pass.

## 5. Authorization boundary

Approval authorizes implementation validation of this corrected candidate filter. It does not
authorize protocol freeze, primary model training, test-split generation, or test-label evaluation.
Until exact-digest owner approval, v1.1 data generation remains blocked.

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
