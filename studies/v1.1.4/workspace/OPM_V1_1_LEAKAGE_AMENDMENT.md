---
document_type: OPM_IMPLEMENTATION_SPEC_AMENDMENT
amendment_version: 1.1.0
base_specification_version: 1.0.0
charter_version: 1.2.0
lifecycle_state: AMENDMENT_APPROVED
approval_status: APPROVED
scientific_core_status: AUTHORIZED_FOR_IMPLEMENTATION_VALIDATION
primary_runs_authorized: false
---

# OPM v1.1 leakage-control amendment candidate

## 1. Purpose and provenance

Canonical OPM v1.0 remains a completed, immutable implementation-validation result whose ORC-005
raw oracle-channel family failed. The renamed-argument-ID channel achieved 0.5098 accuracy on the
canonical validation split, its 95% Wilson interval excluded 0.50, and its exact-binomial p-value
crossed the first Holm boundary. Five post-gate diagnostic replicas showed a weaker recurring upward
bias. No v1.0 artifact, decision rule, or result is superseded by this amendment.

This candidate defines a scientifically distinct v1.1 experiment. It corrects relation balancing for
negative examples and changes the construction of oracle entity-binding IDs. The purpose is to make
relation and argument-ID marginals controlled dataset properties rather than consequences of an
incorrect allocator or finite renderer renaming.

## 2. Normative inheritance

All provisions of `OPM_V1_IMPLEMENTATION_SPEC.md` version 1.0.0 remain normative except where this
amendment explicitly replaces them. In particular, the typed algebra, procedure definitions,
renderers, split sizes, operation/label/relation quotas, model conditions, training schedule, metrics,
five model seeds, claim rules, and primary-run prohibition remain unchanged.

The following provisions are replaced or clarified:

- REN-001A relation and corruption allocation;
- REN-002 oracle `argument_slots` serialization;
- the sentence in REN-003 that makes renderer surface renaming supply canonical endpoint IDs;
- MOD-004 canonical selected-fact endpoint bindings;
- MOD-005 query argument bindings;
- ORC-005 probe 3 feature extraction;
- DAT-003 dataset seed namespaces and DAT-004 fingerprints for v1.1 artifacts.

## 3. Fresh experiment identity

Every v1.1 dataset uses a fresh namespace:

```text
opm-v1.1/<split>/world/<index>
```

No v1.0 world, example, manifest, fitted probe, checkpoint, or prediction may be reused as v1.1
canonical evidence. Code may be reused only after its conformance tests pass against this amendment.
The v1.0 failure and diagnostic artifacts remain archived and reportable.

## 4. Corrected allocation and counterbalanced oracle bindings

### 4.1 Root conformance correction

The v1.0 implementation applied the declared `LOOKUP` and `CHAIN` relation quotas only to positive
cells. This violated REN-001A/DAT-001. For example, canonical validation `PROGRAM/LOOKUP` contained
negative counts `BEFORE=431`, `LINK=188`, `SAME=31`, and `WITHIN=150`, while every positive relation
count was 200. Similar imbalances occurred in all domains. Because query relation constrains argument
types and roles, this defect supplied a proxy to the argument-ID probe.

V1.1 makes the intended rule explicit and executable. Relation quotas apply separately inside every
`(split, domain, operation, label)` cell:

- `LOOKUP`: equal `BEFORE`, `LINK`, `SAME`, and `WITHIN` quotas;
- `CHAIN`: equal `BEFORE` and `SAME` quotas;
- `REVERSE`: equal `LINK` and `SAME` quotas;
- `LIFT`: `WITHIN` only.

Any remainder is assigned in the relation order written above. For a negative relation quota, its
valid corruption types are balanced using the existing floor-and-remainder rule inside that relation,
not across a mixture of relations. Insufficient supply generates additional worlds and ultimately
fails loudly at the declared maximum; it never borrows from another relation or corruption quota.

### 4.2 Separation from surface rendering

Surface rendering remains exactly as specified in v1.0 and continues to use a world-level permutation.
Surface aliases are not oracle entity IDs. The model receives only the counterbalanced binding IDs
defined here for query arguments and selected-fact endpoints.

The binding is scoped to one materialized example. It is a type-preserving bijection over the object
inventory `0..31` and container inventory `32..47`. The same binding is applied consistently to every
query argument and every fact endpoint in that example. No binding is shared between examples, and
the label is never an input to the binding function.

### 4.3 Label-matched schedule

Allocation first produces the exact `(split, domain, operation, label, query_relation)` strata above.
Within each stratum, examples are sorted by the existing canonical candidate rank. Let `j` be the
zero-based rank inside that stratum. Positive and negative strata use the same schedule from `j = 0`;
consequently the two labels receive identical target-ID histograms without consulting probe
predictions.

For a query whose arguments have different types:

```text
object_target(j)    = j mod 32
container_target(j) = 32 + (j mod 16)
```

The target appropriate to each semantic argument role is used. For a query whose two arguments are
objects:

```text
arg1_target(j) = j mod 32
arg2_target(j) = (arg1_target(j) + 1 + floor(j / 32)) mod 32
```

The schedule never maps distinct query objects to the same target. `WITHIN` queries use the
object/container rule. OPM v1.1 contains no container/container query.

For each relation stratum, target frequencies differ by at most one. Because positive and negative
relation quotas and schedules are identical, their marginal binding-ID histograms are exactly
identical for every `(domain, operation, query_relation, argument_position, entity_type)` stratum.

### 4.4 Completing the bijection

After fixing the query-entity targets, assign all remaining entities of each type to unused IDs of the
same type. Rank remaining `(entity_id, target_id)` candidates by:

```text
SHA256("opm-v1.1/binding/<example_id>/<entity_type>/<entity_id>/<target_id>")
```

Traverse latent entities in ascending ID order and give each the lowest-ranked unused target for that
entity. This deterministically completes a bijection. The algorithm must assert type preservation,
injectivity, query-target correctness, and identical positive/negative marginal histograms before
writing a split.

### 4.5 Primitive boundary

`argument_entity_ids` and every pair in `fact_endpoint_ids` use this example-scoped binding. A selected
fact and query that refer to the same latent entity therefore receive the same ID inside the example.
Unselected facts do not expose binding information to a primitive. Tensor shapes and embedding table
sizes remain unchanged.

## 5. Revised raw leakage gate

ORC-005 retains all three probes, hyperparameters, thresholds, Wilson requirements, exact-binomial
tests, and Holm correction from v1.0. Probe 3 is clarified to use serialized
`argument_entity_ids`—the counterbalanced oracle binding—not aliases parsed from rendered text.

The gate is run once on fresh v1.1 canonical train and validation artifacts after all construction
invariants pass. Diagnostic namespaces, test splits, and test labels are not used. Failure blocks v1.1
protocol freeze and may not trigger automatic regeneration, seed search, threshold changes, or another
amendment presented as confirmatory.

## 6. Required conformance evidence

Before v1.1 may enter `PROTOCOL_FROZEN`, implementation validation must show:

1. fresh v1.1 namespaces and zero world/example overlap with v1.0;
2. exact original split, operation, label, relation, and corruption quotas;
3. exact positive/negative relation-count and argument-ID histogram equality in every declared train
   and validation stratum;
4. type-preserving bijections and within-example entity-identity consistency;
5. unchanged tensor shapes, model parameter counts, and active FLOP tolerances;
6. passing canonical ORC-005 results under the unchanged statistical rule;
7. new immutable manifests, hashes, trace fixtures, and traceability records;
8. locked-evaluator controls before any test-label access;
9. an explicit scientific-owner lifecycle decision after reviewing all evidence.

## 7. Prohibited interpretations and actions

- v1.1 may not be described as repairing or passing v1.0.
- The v1.0 canonical failure remains in every comparative report.
- Diagnostic replicas may not enter v1.1 estimates.
- No primary model training or test-label evaluation is authorized by approving this amendment alone.
- Approval authorizes implementation validation and fresh dataset construction only.

## 8. Proposed lifecycle transition

Upon exact-digest owner approval, this amendment may transition to:

```yaml
amendment_version: 1.1.0
lifecycle_state: AMENDMENT_APPROVED
scientific_core_status: AUTHORIZED_FOR_IMPLEMENTATION_VALIDATION
primary_runs_authorized: false
```

This transition is now effective. Implementation agents may implement and validate the amended
allocator and binding algorithm and may generate fresh v1.1 canonical data. Primary model training
and test-label evaluation remain prohibited.

## Approval record

```yaml
approval_status: APPROVED
approved_by: Dwight Robert Keller-Williams
approved_candidate_normative_prefix_sha256: 479fbe9beb4adba342bfe89ab817d9a86ac097aa536365f6fff0c5808937d845
approved_candidate_whole_file_sha256: 43015a93667f34da4b881617ea64e28985a9a9a121d102f59c19186b193e9e66
effective_post_transition_normative_prefix_sha256: c0bf6807463ea15baabfb968ad2816e863b666b8bd3f3bf0ba9911b7af8d5c8d
approval_statement: I, Dwight Robert Keller-Williams approve OPM v1.1 Leakage-Control Amendment version 1.1.0 with normative-prefix SHA-256 479fbe9beb4adba342bfe89ab817d9a86ac097aa536365f6fff0c5808937d845.
approved_scope: v1.1 scientific-core translation and IMPLEMENTATION_VALIDATION
primary_runs_authorized: false
```

The normative-prefix digest is SHA-256 over the exact UTF-8 bytes from the beginning of this file up
to, but excluding, the `## Approval record` heading.

## Post-approval implementation clarifications

These clarifications record the scientific owner's implementation-review conditions. They explain the
approved construction without changing its algorithm, statistical gate, or normative-prefix digest.

1. Label is used only to construct matched counterbalancing strata. It is not supplied to the
   target-ID schedule, bijection-completion hash, model, or probe features. The earlier statement that
   label is never an input refers to those downstream functions, not to stratum construction.
2. Implementation validation must report positive/negative marginal histograms for every selected
   fact position and each endpoint position within that fact, stratified by entity type, fact relation,
   operation, and domain. These are diagnostic construction checks, not additional ORC-005 gates.
3. Joint endpoint patterns must not be forced independent of the label or procedure. Entity matching
   and compositional compatibility are intended task information; only unintended marginal ID cues
   are diagnosed.
4. `PROTOCOL_FROZEN` additionally requires verified candidate-archive provenance, all conformance
   evidence in section 6, and a fresh explicit scientific-owner lifecycle decision.
