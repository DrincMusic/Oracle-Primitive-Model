# OPM v1 implementation-validation status

```yaml
status_record:
  specification_version: 1.0.0
  approved_amendment_version: 1.1.3
  charter_version: 1.2.0
  lifecycle: PROTOCOL_FROZEN
  protocol_frozen: true
  primary_runs_authorized: false
  recorded_on: 2026-08-20
  environment:
    python: 3.12.10
    pytorch: 2.8.0+cu128
    numpy: 2.5.2
    cuda_available: true
    target_device: NVIDIA GeForce RTX 4090
```

Amendment v1.1 is approved for implementation validation with approved candidate normative-prefix
SHA-256 `479fbe9beb4adba342bfe89ab817d9a86ac097aa536365f6fff0c5808937d845` and effective
post-transition normative-prefix SHA-256
`c0bf6807463ea15baabfb968ad2816e863b666b8bd3f3bf0ba9911b7af8d5c8d`.
The exact pre-transition candidate is archived and independently verifies against both approved
candidate hashes. Post-approval clarification records label's limited role in matched-stratum
construction and requires selected-fact endpoint marginal diagnostics before protocol freeze.

The v1.1.1 binding-schedule erratum is approved for implementation validation. It replaces the
collision-producing offset with the modulo-31 schedule and preserves all other v1.1 requirements.
No v1.1 data have yet been generated.

The first full v1.1 validation allocation then stopped before writing data on an endpoint-corrupted
`BEFORE(6,6)` query. Self-endpoint corruptions make argument equality a direct negative-label cue and
cannot be counterbalanced without breaking entity identity. V1.1.2 is now approved and excludes a
replacement equal to argument 1 before ranking and binding. Fresh v1.1 train and validation generation
may resume; test generation and primary training remain prohibited.

Fresh v1.1.2 train and validation artifacts were then generated once under `opm-v1.1` namespaces.
Training contains 144,000 rows with SHA-256
`8c6f994dbdba2450bfe0be60f4231e5f7122e2aa3cbea733e6cfb4f9d45ee998`; validation contains
14,400 rows with SHA-256 `f0319c2fe5cd792169b461b8f965b102b1c6b0ddb3141436985b95e04a7ae100`.
Conformance passes exact relation histograms, exact query-argument histograms, and zero binding
collisions. Selected-fact endpoint diagnostics are recorded and are not treated as gates.

The first v1.1.3 construction is an invalid pre-canonical construction preserved for provenance. It
failed its pre-ORC conformance check because serialization recovered
evidence positions with value-based lookup. For repeated equal-valued evidence facts, this collapsed
two scheduled occurrences onto the first slot. ORC-005 was not run on that construction. The complete
failed construction and report are preserved under
`generated/v1.1.3-failed-pre-serialization-fix/`, including the original train and validation hashes
`506895eddfa67ecc785a46f55065f406ea72ea6ad3a59a6328628dd87255ed8b` and
`3031af6a6cd004cea01d54b31aa5f30aa207cdd79175699b73fb2a92ce7fcdff`.

Following explicit owner authorization, the serializer was corrected to retain scheduled occurrence
indices and the v1.1.3 train and validation artifacts were reconstructed. This reconstruction is the
sole canonical v1.1.3 construction. The reconstructed train set
contains 144,000 rows with SHA-256
`4f2c07bfc0400c992a936fa7b64f3fabefcd2f8b29064bf610a7c83c283deafa`; validation contains 14,400
rows with SHA-256 `1dee006f0db36bf77925f6f375a8d776f4049059c2994d65ee1f1e9eed58236f`.
Conformance passes exact relation and query-argument histograms, zero binding collisions, and exact
positive/negative marginal and joint evidence-position histograms. Endpoint marginals remain recorded
as diagnostics rather than gates. No test data or labels were accessed.

The one-shot v1.1.3 ORC-005 family passes:

| Channel | Accuracy | 95% Wilson interval | Unadjusted p | Gate |
|---|---:|---:|---:|---|
| Operation tokens + step count | 0.5000 | [0.4918, 0.5082] | 0.50332 | PASS |
| Evidence positions | 0.5000 | [0.4918, 0.5082] | 0.50332 | PASS |
| Counterbalanced argument IDs | 0.5000 | [0.4918, 0.5082] | 0.50332 | PASS |

The Holm family and descriptive gates pass. Per the approved erratum, this evidence does not itself
freeze the protocol or authorize primary runs; a fresh scientific-owner decision is required.

The one-shot v1.1.2 ORC-005 family **fails** on evidence positions:

| Channel | Accuracy | 95% Wilson interval | Unadjusted p | Gate |
|---|---:|---:|---:|---|
| Operation tokens + step count | 0.5000 | [0.4918, 0.5082] | 0.50332 | PASS |
| Evidence positions | 0.5116 | [0.5034, 0.5198] | 0.00276 | **FAIL** |
| Counterbalanced argument IDs | 0.5000 | [0.4918, 0.5082] | 0.50332 | PASS |

The evidence-position result crosses the first Holm boundary and its Wilson interval excludes 0.50.
No seed regeneration, fact-order change, threshold change, or automatic retry is authorized.

Train/validation-only diagnosis localizes the v1.1.2 failure to recurring position bias. Step-1 and
step-2 train/validation positive-rate correlations are 0.767 and 0.868. Equal-world accuracy is 0.5119
with world-cluster interval [0.5032, 0.5201]; `CHAIN` accuracy is 0.5441. Both individual evidence
steps leak weakly. This supports a deterministic ordering coupling rather than a fixed validation
fluctuation. No test labels were accessed. `OPM_V1_1_3_EVIDENCE_ORDER_ERRATUM.md` proposes a fresh,
matched evidence-position schedule and is now approved for implementation validation. Fresh v1.1.3
train and validation generation is authorized; test generation and primary training remain blocked.

## Implemented

- Typed entities, facts, queries, and relation-specific complete entailment.
- Deterministic `PCG64DXSM` world generation and seed derivation.
- Positive procedure enumeration and structurally corrupted negative candidates.
- `SET`, `SCENE`, and `PROGRAM` renderer variants 0–2.
- Frozen lexer and generated vocabulary.
- Canonical entity renaming and selected-fact endpoint bindings.
- Eight-fact materialization, deterministic distractors, fact shuffling, and evidence indices.
- Tensor collation with typed entity ordinals.
- Shared, domain-untied, cloned, and domain-generalist neural conditions.
- Fixed per-seed operation-to-module permutation and unused sentinels.
- Fact/query encoders, primitive transition equations, decoder, and one/two-step execution.
- Parameter accounting, bounded validation training, evaluation, ablation, and adapter-only paths.
- Raw logistic leakage-probe and Holm decision machinery.
- CLI environment, fixture, model-smoke, and hard-gated primary-run commands.
- Reduced-scale split allocation with observed/recombination holdout rules.
- Corruption-balanced negative allocation and immutable JSONL manifests.
- Deterministic four-case dry-run trace generation.
- Paired two-level bootstrap implementation with key-set validation.
- CPU forward-FLOP tracing and active-compute comparison.
- Selected-evidence representation extraction and per-step neural-probe execution.
- H1 claim decision and explicitly synthetic claim-ledger validation.
- One-row-per-identifier requirement-to-implementation/test/artifact traceability output covering all
  66 normative identifiers.
- Content-addressed run identities and immutable environment/run manifests.
- Model/optimizer/RNG checkpoints with deterministic resume validation.
- Validation-only input/label separation with hash manifests and access audit logs.
- Fail-closed locked evaluator with lifecycle authorization, fingerprint verification, exact-key
  prediction checks, denial/success auditing, and aggregate-only results.
- Bounded trained-representation probes using the specified 100 probe epochs.
- Fail-closed protocol-freeze readiness reporting.
- Approved-scale allocation with exact relation subquotas and canonical row-count plans.
- Canonical test-input/label separation and six-split contamination auditing.
- Canonical three-channel raw oracle leakage probing with Wilson and Holm gates.

## Validation evidence

```text
Scoped Ruff:       PASS
OPM tests:         47 passed
CLI environment:   PASS
Fixture report:    PASS
Four-model smoke:  PASS (1 optimizer step per model)
Primary-run blocking-control test: PASS
Repository suite:  452 passed, 3 unrelated pre-existing failures
```

Generated validation-only artifacts:

```text
test-recombination.validation.jsonl
  rows:    6
  sha256:  cf5ad430511c8c40e9c6049d043736a834ec6015fe6ca3fd456a597e519cf572

dry-run-traces.json
  traces:  4
  sha256:  ebe38e3cf0b23a2ecefdeff3e23f0b8c84c33ea2b4a0055a4d9a04269bf9aa8a
```

CPU profiler validation for a mixed three-example batch:

| Model | Forward FLOPs | Total parameters |
|---|---:|---:|
| `OPM_SHARED` | 392,765,952 | 4,287,666 |
| `PROC_UNTIED` | 392,765,962 | 9,036,978 |
| `PROC_CLONE` | 392,765,962 | 9,036,978 |
| `DOMAIN_GENERALIST` | 392,950,272 | 2,859,122 |

All validation FLOP ratios relative to `OPM_SHARED` are within the declared ±2% matching tolerance. These are CPU profiler estimates, not H3 hardware evidence.

The neural-probe smoke artifact uses an untrained model, one probe epoch, and relaxed validation thresholds. It validates extraction, per-step/per-run seeding, fitting, and Holm plumbing only. It is not a mechanism result.

The trained-probe validation artifact uses five bounded optimizer steps for one model condition and
seed, followed by the specified 100 probe epochs. Both step probes passed at this reduced scale. This
validates the trained-representation workflow only; it is not the canonical five-seed mechanism test.

The validation label seal contains 36 reduced-scale rows. Inputs and labels have separate hashed
files, and label access is append-audited. This is tooling evidence, not a sealed primary dataset.

The locked evaluator validation harness consumed label-blind constant predictions for those 36 rows,
verified the sealed input and label fingerprints, enforced exact prediction-key coverage, returned
aggregate results only, and appended an authorized access record. Separate tests prove that the
production evaluator requires `PROTOCOL_FROZEN`, explicit sealed-label-access authorization, and
explicit aggregate-test-evaluation authorization simultaneously. Frozen-state tests prove access is
denied when either or both authorizations are false; duplicate prediction keys also fail closed. This
validates the control boundary without generating or accessing any test split.

Traceability now contains 66 unique rows, one for every normative identifier from `ALG-001` through
`SYS-004`. Each row identifies implementation, tests, evidence artifact, and current validation status.

Following explicit owner authorization, all four canonical v1.1.3 primary test splits were generated
once and immediately separated into model-readable inputs and sealed labels:

| Split | Rows | Input SHA-256 | Sealed-label SHA-256 |
|---|---:|---|---|
| Test-interpolation | 14,400 | `c9c7c455a4cb1c7d9f1627b94fa86df99b4a0e016ca9509f163710ef53a27956` | `e4227362e26b05a016632cb0629b6d8aa51114e6c8ff4662c59257fb1bcc7522` |
| Test-recombination | 18,000 | `102a73c6ad367df12d7b0358deff2981ba0720001453eb686b18ce692728f823` | `3acfd686311bbe1318fe8b325d6229891f63839adef8cd12ef551795f3cbb959` |
| Test-renderer | 9,600 | `9437767f6f3e9950aa6f32c48a414ea39572f191315ef76d9b63024948f078d0` | `a0c317849e3c7d48e511a3204a1510bc585d483fc139b1b0cedc259ecaef397c` |
| Test-structural | 9,600 | `5730f70dfab13f55e708eae426beda7baabb4771aa7d7ae29a05161a4cf9ad50` | `781871ee5334a9a58edd8138a313e7a33c25acbc4f224ef38825bcb732294fc0` |

The label-blind six-split audit covers 210,000 rows, 6,700 unique worlds, and 210,000 unique
cross-split latent fact/query records. It passes input hashes, counts, renderer/holdout policy, and
world/latent contamination checks. It verifies seal presence and recorded label fingerprints without
opening sealed-label content. The creation audit records no post-sealing label access, evaluation, or
training. No aggregate test result has been computed.

The post-transition readiness audit reports `ready_for_protocol_freeze: true` and
`primary_runs_authorized: false`. Every pre-freeze check and the exact-digest owner transition now
pass; post-freeze experimental authority remains separately withheld. Canonical trained-representation
probes require primary trained representations and therefore remain post-freeze experimental work.

Target-hardware validation uses the local NVIDIA GeForce RTX 4090 with PyTorch `2.8.0+cu128`, CUDA
runtime 12.8, cuDNN 9.1.0, compute capability 8.9, and 25,756,696,576 bytes of reported device memory.
All four conditions completed bounded CUDA profiler smoke traces on the same three-example synthetic
fixture. Forward-FLOP counts range from 392,765,952 to 392,950,272, remaining within the declared
matching tolerance. Median smoke latencies range from 5.02 ms to 6.45 ms; these timings are descriptive
implementation-validation measurements and are not canonical H3 evidence. CUDA checkpoint/recovery
exactly reproduced the uninterrupted next optimizer step after correcting device-specific RNG
restoration. No canonical data,
test data, primary training, or aggregate evaluation was used.

All six v1.0 dataset artifacts were generated without running training, but their earlier audit is now
**invalidated** by a relation-quota conformance defect:

| Split | Rows | Input SHA-256 |
|---|---:|---|
| Train | 144,000 | `cf6f840def0521cdce7e4516a63c7a7bda9f5874678aa1828799aa0d729a8902` |
| Validation | 14,400 | `08ad98efc381cdaa08ff48da66c412ae346058f2d7d8d79c6261dfe7da668e81` |
| Test-interpolation | 14,400 | `ac59573f28dc4de931fbf0aea6f66376d376f67b2fb2062ee8370f42e4768371` |
| Test-recombination | 18,000 | `3446b5f616b94c2c992949cf636bae1a8dd51310cef30d37c5431720d35d87c7` |
| Test-renderer | 9,600 | `57c07bfbde760898bb094078b7b0870da53af838bd7937e61efbb0cb5e485357` |
| Test-structural | 9,600 | `d9744fb0c8b678d355450f7be3f083bd8856b75dc17e4725897ec634ef551d31` |

The original audit covers 210,000 rows, 12,018 unique worlds, hashes, holdout policy, renderer policy,
and cross-split latent uniqueness. It incorrectly treated operation/label counts as sufficient and did
not verify negative relation subquotas. Test labels remain absent from all
model-readable input JSONL files and stored separately with hashes. They are not yet access-controlled
by a locked evaluator, so this does not by itself satisfy the primary-label sealing gate.

The negative allocator balanced corruption types across the entire operation cell before balancing
query relations. This violates the approved `LOOKUP` and `CHAIN` relation composition. In validation,
for example, `PROGRAM/LOOKUP/negative` is `BEFORE=431, LINK=188, SAME=31, WITHIN=150`, versus exactly
200 each for positives. The defect provides a relation/type proxy to the argument-ID probe and is a
credible root cause of the observed signal. Therefore the generated v1.0 data are not
specification-conformant canonical evidence, and their passing audit record is superseded by
`canonical-data.audit.invalidation.json`.

The canonical ORC-005 raw leakage family **fails** because one of three channels leaks label signal:

| Channel | Accuracy | 95% Wilson interval | Unadjusted p | Gate |
|---|---:|---:|---:|---|
| Operation tokens + step count | 0.5000 | [0.4918, 0.5082] | 0.50332 | PASS |
| Evidence positions | 0.5031 | [0.4949, 0.5112] | 0.23423 | PASS |
| Within-world renamed argument IDs | 0.5098 | [0.5016, 0.5180] | 0.00960 | **FAIL** |

The smallest p-value is below the first Holm threshold, `0.05 / 3 = 0.01667`, and its Wilson
interval does not contain 0.50. The implementation therefore records a genuine pre-freeze blocker.
No post-result feature, threshold, generator, or allocation tuning has been performed.

Post-gate diagnosis used only canonical train and validation data; no test labels were accessed.
The combined argument probe is driven more by `PROGRAM` (0.5179) than `SCENE` (0.5063) or `SET`
(0.5052), and more by `LOOKUP` (0.5140) and `CHAIN` (0.5144) than the other operations. Argument 1
alone scores 0.5042 and argument 2 alone scores 0.5046. Equal-weighting the 709 validation worlds
gives 0.5096, with a deterministic world-cluster bootstrap interval of [0.5017, 0.5180]. The
train-to-validation correlation of per-alias label rates is 0.113 for argument 1 and 0.373 for
argument 2. This localizes the signal but does not yet distinguish a fixed-dataset realization from
a deterministic generator/allocation coupling.

Five canonical-sized, noncanonical validation replicas were then generated under explicit diagnostic
namespaces, with zero canonical-world overlap and no test-label access. Their argument-probe
accuracies were 0.5049, 0.5103, 0.5065, 0.5042, and 0.5016 (mean 0.5055). One of five crossed the
canonical first-Holm boundary. All five were above chance, and argument-2 train/replica alias-rate
correlations were positive in every replica (0.206 to 0.445). Thus gate-level failure is not reliably
reproduced, but a weak recurring allocation/renaming coupling is more plausible than a purely isolated
canonical fluctuation. These post-gate replicas are diagnostic only and cannot rescue or replace the
failed canonical result.

The three repository failures occur in existing observer-chat tests because calls to
`ObserverChat._session` omit a newly required `profile_id`. No OPM implementation file appears
in their traces, and this validation did not modify that unrelated subsystem.

## Protocol-freeze transition

The scientific owner approved the exact pre-transition status artifact with SHA-256
`ee6971a94ab219dfd6baf1aad3fe6c4dfdc81dece80300cb350ebda9b788d6c6` and authorized transition to
`PROTOCOL_FROZEN`. The exact approved candidate is archived as
`OPM_V1_STATUS_PROTOCOL_FREEZE_APPROVAL_CANDIDATE.md`. Primary runs remain separately unauthorized.

## Post-freeze experimental work

- All-condition, five-seed primary training.
- Canonical neural evidence-vector MLP probes for every condition and seed.
- Canonical H1, H2, and H3 evaluation; H4 remains excluded as specified.
- Locked aggregate test evaluation without exposing sealed labels.
- Final statistical and claim decisions.

## Next lifecycle action

Obtain separate explicit owner authorization before beginning primary training, canonical trained
probes, sealed-label access, aggregate test evaluation, or claim decisions.

## Protocol-freeze approval record

```yaml
approval_status: APPROVED
approved_by: Dwight Robert Keller-Williams
approved_candidate_sha256: ee6971a94ab219dfd6baf1aad3fe6c4dfdc81dece80300cb350ebda9b788d6c6
authorized_transition: PROTOCOL_FROZEN
primary_runs_authorized: false
sealed_label_access_authorized: false
aggregate_test_evaluation_authorized: false
claim_decisions_authorized: false
approval_statement: I, Dwight Robert Keller-Williams, approve OPM v1.1.3 implementation-validation status with SHA-256 ee6971a94ab219dfd6baf1aad3fe6c4dfdc81dece80300cb350ebda9b788d6c6 and authorize transition to PROTOCOL_FROZEN. This transition does not itself authorize primary training, trained probes, sealed-label access, aggregate test evaluation, or claim decisions.
```
