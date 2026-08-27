# OPM v1.1.4 final study report

Generated: `2026-08-27T21:34:48.792452+00:00`  
Closeout authorization: `4a87ff153338c18ecb8d6992a530bf5a80e7401dceefbefa8e0d55e03ade9e42`  
Authoritative claim-decision freeze: `2c14678accac56e6708ad12813314aad835cd72a02e240111b8490b8086f34a5`  
Claim-decision Merkle root: `6a237d348ef4d07d48d8487f1e67d18d01a910b4854f76b200631ac0b0e4c886`

## Executive conclusion

OPM v1.1.4 provided preregistered, sealed-test evidence that shared primitive organization produces a large recombination-generalization advantage without materially degrading interpolation performance. H1 was **SUPPORTED**. `OPM_SHARED` exceeded `DOMAIN_GENERALIST` by `0.426200` (42.6200 percentage points), with a 95% bootstrap interval of [30.1938, 49.2272] percentage points, and passed both frozen interpolation non-inferiority checks.

The complete preregistered composite mechanism was **NOT_SUPPORTED**. The canonical neural-probe criterion failed for OPM seed 4404, evidence step 1, while symbolic-oracle and raw ORC-005 inputs were unavailable in aggregate v4. The functional outcome and mechanism conclusion are therefore kept separate.

## Authority and scope

Formal statuses and their observed values are reproduced without modification from the frozen claim-decision package. Descriptive aggregate tables come from aggregate v4 files transitively bound by that claim authorization. Pilot and primary validation records are developmental context only and do not enter claim statistics.

Status inventory: **6 SUPPORTED**, **2 NOT_SUPPORTED**, **0 INCONCLUSIVE**, and **7 NOT_EVALUABLE**.

No checkpoint, model, prediction row, sealed label, or raw representation was read during closeout. No training, prediction generation, aggregation, threshold application, threshold change, claim-status change, or new scientific decision occurred.

## Preregistered hypotheses and thresholds

| Identifier | Exact preregistered wording | Frozen comparison rule |
|---|---|---|
| `H1-PRIMARY` | Reusable primitives improve performance on valid, withheld domain-operation combinations or unseen procedural compositions relative to suitable non-sharing and generic baselines. | SUPPORTED iff lower95(Delta_generalist)>0.02 and OPM interpolation is no more than 0.01 below DOMAIN_GENERALIST and PROC_UNTIED; NOT_SUPPORTED iff upper95<=0.02; otherwise INCONCLUSIVE |
| `H2-THEORETICAL-RESOURCE-ADVANTAGE` | At matched quality, the sparse primitive architecture uses meaningfully fewer active parameters, active FLOPs, or stored duplicate parameters than an appropriate dense or non-sharing alternative. | STA-005 has no preregistered quality-matched resource threshold; NOT_EVALUABLE |
| `H3-REAL-HARDWARE-ADVANTAGE` | The theoretical savings produce measurable improvements in wall-clock latency, throughput, training time, memory, or energy on specified target hardware. | STA-006 declares H3 out of scope; NOT_EVALUABLE |
| `H4-CORRECT-BOUNDARY-OF-REUSE` | The architecture transfers rules that are genuinely shared without catastrophically transferring rules that are domain-specific. | MET-008 and STA-007 declare H4 out of scope; NOT_EVALUABLE |
| `MECHANISM-CAUSALLY-REUSABLE-PRIMITIVE` | Evidence of reusable computation additionally requires all of: | SUPPORTED only if every STA-004 criterion is SUPPORTED |
| `MET-005-INTERCHANGE-MECHANISM` | Pair examples from the same latent world, query, label, and procedure but different renderers. At the boundary before the last primitive call, swap normalized execution states while retaining the destination evidence vector. Report swapped accuracy and the drop from unswapped paired accuracy. | no preregistered numerical decision threshold; NOT_EVALUABLE |
| `MET-007-SURFACE-INVARIANCE` | Use renderer variant 2 with relation symbols reassigned through a test-only permutation. Oracle bindings remain correct. Report the accuracy drop relative to the ordinary renderer test. | no preregistered numerical decision threshold; NOT_EVALUABLE |
| `STA-004-ACTIVE-ABLATION` | ablating the corresponding active primitive reduces its selected-operation accuracy by at least `0.20` in at least two trained domains; | for every active operation, count trained interpolation domains with baseline_accuracy-intervention_accuracy >= 0.20; require count >= 2 |
| `STA-004-ADAPTER-ONLY` | adapter-only accuracy is at most `0.60`; | maximum OPM_SHARED adapter-only split accuracy <= 0.60 |
| `STA-004-NEURAL-PROBES` | and neural evidence-vector probes pass ORC-006 for every run used in the mechanism claim. | for each OPM_SHARED seed, both accuracies <=0.55, both Wilson intervals contain 0.50, and two-test Holm at alpha=0.05 rejects neither one-sided null |
| `STA-004-RAW-ORACLE-PROBES` | raw oracle leakage probes pass ORC-005; | requires all three ORC-005 results and Holm family decision |
| `STA-004-RECOMBINATION` | shared-model recombination accuracy exceeds `0.80`; | OPM_SHARED pooled recombination accuracy > 0.80 |
| `STA-004-SENTINEL` | sentinel ablation changes all logits by less than `1e-7`; | maximum of all frozen sentinel max-absolute-logit-change values < 1e-7 |
| `STA-004-SYMBOLIC-ORACLE` | symbolic oracle accuracy equals `1.000`; | observed accuracy == 1.000 |
| `STA-004-UNRELATED-PRESERVATION` | average unrelated-operation accuracy drop under that ablation is below `0.05`; | for every active operation, arithmetic mean of baseline_accuracy-intervention_accuracy over unrelated interpolation domain-operation aggregate rows < 0.05 |

`Delta_untied` is descriptive and has no independent H1 support threshold. MET-005 and MET-007 specify reported metrics but no numerical decision thresholds. H2 requires a later quality-matched resource threshold; H3 and H4 are out of scope in v1.

## Pilot and primary methodology

The experiment used three rendered domains (`PROGRAM`, `SCENE`, and `SET`) and four operation families (`LOOKUP`, `REVERSE`, `CHAIN`, and `LIFT`). Training withheld `SET × REVERSE`, `SCENE × LIFT`, and `PROGRAM × CHAIN`; the primary recombination test contained only those equally weighted cells.

The primary conditions were `OPM_SHARED`, `DOMAIN_GENERALIST`, `PROC_UNTIED`, and `PROC_CLONE`. Pilot seed 1101 evaluated all six combinations of learning rate `{0.0001, 0.0003, 0.0006}` and dropout `{0.0, 0.1}` for all four conditions: 24 equal-budget 50,000-step runs. The preregistered tie rule selected learning rate `0.0006` and dropout `0.0`. Pilot runs were excluded from claim statistics.

The selected configuration was frozen for 20 primary runs: four conditions × seeds 1101, 2202, 3303, 4404, and 5505. Every run completed 50,000 steps. Checkpoints were selected by observed-cell macro validation accuracy; sealed tests were not used for training or selection. Label-blind Stage 1 generated predictions, interventions, and neural probes. The separately authorized aggregate-v4 stage joined frozen outputs to sealed targets without loading models, then ran the registered 10,000-replicate paired two-level bootstrap. The claim stage applied thresholds separately afterward.

## Developmental pilot and validation results

These results selected configuration/checkpoints only. They are **developmental**, were not sealed-test endpoints, and do not independently support H1 or any mechanism claim.

| Learning rate | Dropout | Four-condition pilot mean | In tie set |
|---:|---:|---:|---|
| 0.0001 | 0.0 | 0.994288 | no |
| 0.0001 | 0.1 | 0.990104 | no |
| 0.0003 | 0.0 | 0.997483 | no |
| 0.0003 | 0.1 | 0.991094 | no |
| 0.0006 | 0.0 | 0.998542 | yes |
| 0.0006 | 0.1 | 0.999132 | yes |

Primary selected-checkpoint macro validation accuracy:

| Condition | Seed | Selected step | Developmental validation accuracy |
|---|---:|---:|---:|
| `OPM_SHARED` | 1101 | 32500 | 0.999306 |
| `OPM_SHARED` | 2202 | 30500 | 0.998958 |
| `OPM_SHARED` | 3303 | 44500 | 0.999583 |
| `OPM_SHARED` | 4404 | 35000 | 0.999722 |
| `OPM_SHARED` | 5505 | 28500 | 0.999444 |
| `DOMAIN_GENERALIST` | 1101 | 41000 | 0.997292 |
| `DOMAIN_GENERALIST` | 2202 | 42000 | 0.996597 |
| `DOMAIN_GENERALIST` | 3303 | 43500 | 0.997847 |
| `DOMAIN_GENERALIST` | 4404 | 30500 | 0.998889 |
| `DOMAIN_GENERALIST` | 5505 | 42000 | 0.997639 |
| `PROC_UNTIED` | 1101 | 28500 | 0.998819 |
| `PROC_UNTIED` | 2202 | 29500 | 0.998889 |
| `PROC_UNTIED` | 3303 | 44000 | 0.996667 |
| `PROC_UNTIED` | 4404 | 35500 | 0.999306 |
| `PROC_UNTIED` | 5505 | 33500 | 0.996181 |
| `PROC_CLONE` | 1101 | 33500 | 0.998750 |
| `PROC_CLONE` | 2202 | 30500 | 0.996875 |
| `PROC_CLONE` | 3303 | 49000 | 0.997361 |
| `PROC_CLONE` | 4404 | 30500 | 0.998681 |
| `PROC_CLONE` | 5505 | 34000 | 0.998958 |

## Authoritative aggregate-v4 results

Pooled sealed-test accuracy across all five declared model seeds:

| Condition | Interpolation | Recombination | Renderer | Structural |
|---|---:|---:|---:|---:|
| `OPM_SHARED` | 0.999347 | 0.997811 | 0.999354 | 0.998979 |
| `DOMAIN_GENERALIST` | 0.997264 | 0.571611 | 0.590729 | 0.890854 |
| `PROC_UNTIED` | 0.997583 | 0.499989 | 0.738417 | 0.873667 |
| `PROC_CLONE` | 0.997861 | 0.500000 | 0.749292 | 0.873854 |

| Frozen paired effect | Point estimate | 95% percentile interval |
|---|---:|---:|
| `Delta_generalist` | 0.426200 | [0.301938, 0.492272] |
| `Delta_untied` | 0.497822 | [0.493535, 0.501668] |

Frozen neural-probe means are descriptive and are not substitutes for the per-run ORC-006 Wilson/Holm decisions:

| Condition | Evidence step 1 | Evidence step 2 |
|---|---:|---:|
| `OPM_SHARED` | 0.503031 | 0.496344 |
| `DOMAIN_GENERALIST` | 0.503313 | 0.503219 |
| `PROC_UNTIED` | 0.499219 | 0.500656 |
| `PROC_CLONE` | 0.500844 | 0.502844 |

## Frozen claim decisions

| Claim | Observed evidence | Threshold | Status |
|---|---|---|---|
| `H1-PRIMARY` | Δgeneralist=0.426200; 95% CI [0.301938, 0.492272]; both interpolation checks pass | `{"delta_generalist_lower_bound_gt":0.02,"interpolation_margin":0.01}` | **SUPPORTED** |
| `H2-THEORETICAL-RESOURCE-ADVANTAGE` | Required input unavailable or hypothesis out of scope | `None preregistered` | **NOT_EVALUABLE** |
| `H3-REAL-HARDWARE-ADVANTAGE` | Required input unavailable or hypothesis out of scope | `None preregistered` | **NOT_EVALUABLE** |
| `H4-CORRECT-BOUNDARY-OF-REUSE` | Required input unavailable or hypothesis out of scope | `None preregistered` | **NOT_EVALUABLE** |
| `MECHANISM-CAUSALLY-REUSABLE-PRIMITIVE` | STA-004-ACTIVE-ABLATION=SUPPORTED; STA-004-ADAPTER-ONLY=SUPPORTED; STA-004-NEURAL-PROBES=NOT_SUPPORTED; STA-004-RAW-ORACLE-PROBES=NOT_EVALUABLE; STA-004-RECOMBINATION=SUPPORTED; STA-004-SENTINEL=SUPPORTED; STA-004-SYMBOLIC-ORACLE=NOT_EVALUABLE; STA-004-UNRELATED-PRESERVATION=SUPPORTED | `{"operator":"all","required_status":"SUPPORTED"}` | **NOT_SUPPORTED** |
| `MET-005-INTERCHANGE-MECHANISM` | accuracy=0.346271; change=-0.652895 | `None preregistered` | **NOT_EVALUABLE** |
| `MET-007-SURFACE-INVARIANCE` | accuracy=0.999354; change=0.000000 | `None preregistered` | **NOT_EVALUABLE** |
| `STA-004-ACTIVE-ABLATION` | LOOKUP=3 qualifying domains; REVERSE=2 qualifying domains; CHAIN=2 qualifying domains; LIFT=2 qualifying domains | `{"minimum_drop":0.2,"minimum_trained_domains":2}` | **SUPPORTED** |
| `STA-004-ADAPTER-ONLY` | maximum split accuracy=0.500125 | `{"operator":"<=","value":0.6}` | **SUPPORTED** |
| `STA-004-NEURAL-PROBES` | failed run(s): 4404 | `{"accuracy_at_most":0.55,"familywise_alpha":0.05,"holm_family_size":2,"wilson_contains":0.5}` | **NOT_SUPPORTED** |
| `STA-004-RAW-ORACLE-PROBES` | Required input unavailable or hypothesis out of scope | `{"accuracy_at_most":0.525,"holm_familywise_alpha":0.05,"wilson_contains":0.5}` | **NOT_EVALUABLE** |
| `STA-004-RECOMBINATION` | OPM_SHARED recombination accuracy=0.997811 | `{"operator":">","value":0.8}` | **SUPPORTED** |
| `STA-004-SENTINEL` | maximum=0.0; summaries=80 | `{"operator":"<","value":1e-07}` | **SUPPORTED** |
| `STA-004-SYMBOLIC-ORACLE` | Required input unavailable or hypothesis out of scope | `{"equals":1.0}` | **NOT_EVALUABLE** |
| `STA-004-UNRELATED-PRESERVATION` | LOOKUP mean drop=0.000000; REVERSE mean drop=0.000000; CHAIN mean drop=0.000000; LIFT mean drop=0.000000 | `{"operator":"<","value":0.05}` | **SUPPORTED** |

## Behavioral and mechanism separation

H1 is a behavioral outcome and is supported independently of mechanism status. The controls learned interpolation nearly perfectly, while `OPM_SHARED` preserved near-ceiling recombination performance and the controls degraded. Several component criteria were supported: recombination accuracy, active ablations, unrelated-operation preservation, the sentinel numerical-null criterion, and adapter-only performance.

The phrase **causally reusable primitive** is not supported for v1.1.4. OPM seed 4404, evidence step 1 had validation accuracy 0.513750, a Wilson interval excluding 0.50 on the high side, and one-sided p=0.014349, which rejected at the first Holm threshold 0.025. Symbolic-oracle accuracy and raw ORC-005 results were also unavailable to the aggregate-v4 claim input. A strong H1 result cannot rescue those mechanism criteria, and the mechanism result does not erase H1.

Interchange accuracy was 0.346271 with change -0.652895; surface reversal accuracy was 0.999354 with change 0.000000. Both remain descriptive because the preregistration defined no numerical claim thresholds for them.

## Deviations, unavailable inputs, and diagnostic attempts

| Record | Disposition |
|---|---|
| `v1.1-leakage-amendment` | v1.0 immutable; v1.1 introduced corrected fresh data and leakage gates |
| `v1.1.1-object-binding` | corrected an object-binding schedule arithmetic defect before protocol freeze |
| `v1.1.2-self-endpoint` | corrected a self-endpoint corruption edge case before protocol freeze |
| `v1.1.3-evidence-position` | corrected evidence-position construction coupling; fresh artifacts and unchanged ORC-005 gate |
| `v1.1.4-training-execution` | resolved source-snapshot, sampler, and equal-budget pilot-selection execution ambiguities |
| `post-primary-stage1-diagnostics` | authorization/output attempts v1-v4 failed before a complete freeze; v5 is authoritative |
| `aggregate-v1-v2` | failed closed before aggregate publication; diagnostic only |
| `aggregate-v3` | superseded because independent review found incorrect equal weighting of variable-row worlds in the bootstrap point estimator |

Unavailable or unthresholded items:

- symbolic-oracle accuracy was not present in aggregate v4.
- raw ORC-005 oracle-probe results were not present in aggregate v4.
- MET-005 interchange had no preregistered numerical decision threshold.
- MET-007 surface reversal had no preregistered numerical decision threshold.
- H2 had no preregistered quality-matched resource threshold.
- H3 and H4 were out of scope.

All corrected predecessor artifacts and failed attempts remain preserved. They are provenance, not interchangeable scientific evidence. Aggregate authorization/output versions v1-v3 are explicitly excluded from every formal decision.

## Authorization and freeze chain

| Lifecycle record | SHA-256 / Merkle identity |
|---|---|
| v1.1.4 training transition | `ec597e19bfc299a5c890bcddd46e4daf95fcb034081d974063eeb0889385b5c4` |
| Completed pilot matrix | `ad90a1097bbfe4e950f8db2a123ae742fb0eb400b8d0dd72f4d07a419783d5be` |
| Completed primary matrix | `79107e31d9faee4d8f42685f801b63b503970dace55879de3c087f506338dc82` |
| Post-primary Stage 1 transition v5 | `ddfc05137e09a4020560c20ac3e2fe8ac6fcaa8bdf86d4b2947007f0e26f1da6` |
| Post-primary Stage 1 freeze | `5af3c1243e8cf16c9b5f74dc2ee89412b6e7507fce189f774f6fb835ef65d8ba`; Merkle `74894ded7e5af418da88cd0f80f40068812ed33091da4d53dad2eaa72e4dbe4e` |
| Sealed aggregate authorization v4 | `c41bb1e61655891ea7f47b4ecdada1cddfdfb4f10bf121dc4a0f12b6fc93ce0c` |
| Sealed aggregate freeze v4 | `e9b3c8bd9007ecc52e46ad985065acfff4b1d22d7530bf67d228c964cc1c972a`; Merkle `2d1fa74c1637f621a748ab69f117d9651fed2facae8c88f919419eae8bea1a9b` |
| Claim-decision authorization | `1c9afdcaf96847be7a57e3f75fcd32f22537bc25ab2fe76e15ce8f01670685c9` |
| Claim-decision freeze | `2c14678accac56e6708ad12813314aad835cd72a02e240111b8490b8086f34a5`; Merkle `6a237d348ef4d07d48d8487f1e67d18d01a910b4854f76b200631ac0b0e4c886` |
| Study-closeout authorization | `4a87ff153338c18ecb8d6992a530bf5a80e7401dceefbefa8e0d55e03ade9e42` |

The closeout freeze containing this report is the final identity for the reporting stage and is recorded outside this self-referential report in `OPM_V1_1_4_STUDY_CLOSEOUT_FREEZE.json`.

## Reproducibility

Verification does not require model execution or access to sealed labels. From the repository root:

```powershell
$opmTests = (Get-ChildItem tests -Filter 'test_opm_*.py' | Sort-Object Name).FullName
& .venv\Scripts\python.exe -m pytest @opmTests -q
& .venv\Scripts\python.exe -m scripts.opm_study_closeout_executor verify `
  --workspace . --freeze <path-to-closeout-freeze> --freeze-sha256 <published-sha256>
```

The verifier rehashes every frozen report artifact, recomputes the closeout Merkle root, checks the claim freeze identity, verifies all bound report inputs, and confirms the zero-access/zero-modification guards. Reproducing a new experiment is not the same as verifying this immutable result and requires a new authorization and study identity.

## Limitations and separately preregistered follow-up studies

- `OPM-MECH-COMPLETE-001`: bind symbolic-oracle and raw ORC-005 evidence explicitly and replicate neural probes under a new mechanism preregistration.
- `OPM-MECH-INTERCHANGE-001`: preregister context-aware interchange hypotheses, positive controls, and numerical thresholds.
- `OPM-RESOURCE-001`: preregister a quality-matched H2 parameter/FLOP threshold and full executed-graph accounting.
- `OPM-HARDWARE-001`: preregister H3 target hardware, repeat counts, latency, throughput, memory, and energy endpoints.
- `OPM-BOUNDARY-001`: preregister H4 controlled exceptions and acceptable negative-transfer thresholds.

These are prospective identifiers only. They do not amend, reopen, or reinterpret OPM v1.1.4.

## Final conclusion

OPM v1.1.4 established the preregistered behavioral phenomenon: shared primitive organization delivered a large sealed recombination advantage without materially degrading interpolation performance. It did not establish the complete preregistered causal explanation. In concise terms: **OPM works according to H1; v1.1.4 does not yet completely establish why it works.**
