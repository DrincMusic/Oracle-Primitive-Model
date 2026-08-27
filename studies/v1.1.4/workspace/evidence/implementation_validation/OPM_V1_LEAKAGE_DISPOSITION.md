# OPM v1 ORC-005 leakage disposition

```yaml
artifact_type: DIAGNOSTIC_DISPOSITION_PROPOSAL
specification_version: 1.0.0
lifecycle: IMPLEMENTATION_VALIDATION
canonical_gate_status: FAILED
protocol_freeze_authorized: false
primary_runs_authorized: false
owner_decision_required: false
owner_disposition: B
test_labels_accessed: false
```

## Established facts

The canonical renamed-argument-ID probe scored 0.5098 on 14,400 validation examples. Its two-sided
95% Wilson interval was [0.5016, 0.5180], and its exact one-sided p-value of 0.00960 crossed the first
Holm threshold of 0.01667. ORC-005 therefore failed under the approved rule.

Five independent, canonical-sized diagnostic replicas scored 0.5049, 0.5103, 0.5065, 0.5042, and
0.5016. Only one replica crossed the same numerical boundary, but all five estimates were above 0.50
and all showed positive argument-2 train/replica alias-rate alignment. The evidence supports a weak
recurring coupling rather than a reliably recurring gate-level effect.

The replicas are post-gate diagnostics. They cannot replace, average with, or invalidate the canonical
result.

## Available dispositions

### A. Close canonical OPM v1 as gate-failed

Preserve specification 1.0.0 and all artifacts, record that implementation validation found oracle
channel leakage, and do not enter `PROTOCOL_FROZEN`. This is the cleanest confirmatory treatment of
the approved experiment.

### B. Authorize a new amended experiment

Archive v1 unchanged, draft a new specification version, and make argument-ID balance a construction
invariant rather than a post-generation acceptance test. A defensible amendment would define a
label-blind counterbalancing algorithm before creating fresh train and validation namespaces, freeze
new hashes, and rerun all three raw gates. The amended experiment must be reported as distinct from
the failed canonical v1.

No exact counterbalancing algorithm is approved by this proposal. It requires scientific-owner review
before implementation.

### C. Relax or replace the statistical gate

Changing the threshold, statistical unit, correction family, or Wilson requirement after observing the
failure would directly weaken a preregistered safeguard. This option is not recommended unless framed
as a separately versioned exploratory experiment; it cannot rehabilitate canonical v1.

## Recommendation

Choose A if preserving a clean confirmatory record is the priority. Choose B if the research program
should continue: retain the failed v1 as evidence, create a separately approved v1.1 amendment with
construction-level counterbalancing, and generate entirely fresh canonical namespaces. Do not choose C
as a retroactive correction.

## Owner disposition

Dwight selected disposition B. A separate `OPM_V1_1_LEAKAGE_AMENDMENT.md` candidate has been drafted.
The owner subsequently approved amendment version 1.1.0 with candidate normative-prefix SHA-256
`479fbe9beb4adba342bfe89ab817d9a86ac097aa536365f6fff0c5808937d845`. V1.1 implementation
validation is authorized; primary runs remain prohibited.
