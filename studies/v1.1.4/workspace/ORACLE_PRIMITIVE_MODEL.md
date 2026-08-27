# Oracle Primitive Model

## Persistent research and design specification

**Status:** Normative research charter; OPM v1 is approved for implementation validation, with primary runs still gated  
**Purpose:** Preserve the complete Oracle Primitive Model idea throughout model design, implementation, evaluation, and later replacement of oracle information with learned mechanisms.  
**Initial scope:** A small, controlled mechanism experiment. It is not initially a language model, a production architecture, or a claim about natural-language understanding.

```yaml
document_control:
  document_id: OPM-CHARTER
  document_role: normative_research_charter
  charter_version: 1.2.0
  implementation_target: OPM-v1
  current_lifecycle_state: SPEC_APPROVED
  scientific_core_status: AUTHORIZED_FOR_IMPLEMENTATION_VALIDATION
  blocking_document: OPM_V1_IMPLEMENTATION_SPEC.md
  permitted_work:
    - meaning_neutral_infrastructure
    - approved_scientific_core_translation
    - implementation_validation
  prohibited_work:
    - primary_experiment_execution
  stage_a_routing: fixed_deterministic_per_seed_one_to_one_permutation
  source_of_truth_priority:
    - approved_protocol_amendments
    - approved_OPM_V1_IMPLEMENTATION_SPEC
    - this_charter
    - machine_readable_configuration
    - source_code_defaults
```

If this status block conflicts with prose, the conflict is a specification defect and work on the affected scientific component must stop until the document is corrected. The block does not override the detailed requirements below; it summarizes them for agents and automation.

### Document authority and readiness

This document is the persistent authority for the research question, hypotheses, causal claims, experimental logic, design invariants, and required controls. It is intentionally **not yet the executable implementation specification** for the scientific core.

Until the companion `OPM_V1_IMPLEMENTATION_SPEC.md` is approved, an implementation agent may build only meaning-preserving infrastructure such as:

- repository and package structure;
- configuration and schema validation;
- deterministic seed management;
- experiment tracking and artifact manifests;
- abstract model, renderer, intervention, and metric interfaces;
- FLOP and parameter-accounting utilities;
- baseline scaffolding that does not assume unresolved tensor contracts;
- and tests of infrastructure behavior.

An agent must not independently choose the relational algebra, renderer semantics, primitive-call contract, oracle encoding, scientific loss definitions, model matching, data splits, or success thresholds and then represent the resulting experiment as the canonical OPM v1 experiment.

The implementation-readiness test is:

> Can an implementer translate every scientific requirement directly into code or configuration without choosing what the experiment means?

The answer must be yes before scientific-core implementation begins.

### Required response from an implementation agent

An agent asked to implement the canonical OPM scientific core while `scientific_core_status` is not `AUTHORIZED_FOR_IMPLEMENTATION_VALIDATION` must stop and report the active blocker. It must respond in substance appropriate to that blocker; while the implementation specification is missing or owner approval remains pending, the response is:

> Canonical OPM scientific-core implementation is blocked pending an approved `OPM_V1_IMPLEMENTATION_SPEC.md`. I may build meaning-neutral infrastructure or help resolve and document artifacts A–L, but I must not invent the experiment's algebra, renderers, tensor contracts, oracle encoding, losses, protocol, or thresholds.

The agent should then identify the unresolved artifact or propose work on the companion specification. It must not treat urgency, autonomy, or a general instruction to “build the model” as permission to bypass this gate. Only explicit approval of a conforming companion specification changes the status.

---

## 1. Executive summary

The Oracle Primitive Model (OPM) tests whether a neural network can perform a capability through a sparse, reusable computational basis rather than through separate domain-specific implementations.

The governing thesis is:

> Learn reusable computational primitives, compose them into procedures, and preserve only the domain-specific differences that causally matter.

The word **oracle** means that early experiments supply some parts of the task decomposition that a mature system would eventually need to infer. Depending on the experimental stage, the oracle may provide correct argument bindings, abstract operation or procedure tokens, and execution order. The oracle must never provide the target answer or information from which the answer can be recovered without processing the task instance.

The OPM is an upper-bound mechanism experiment. Its first purpose is to answer:

> If the correct decomposition were available, would procedural reuse actually improve compositional transfer or efficiency?

Only after this mechanism succeeds should routing, binding, parsing, and procedure construction become learned components.

The model is not required to discover a human-readable internal ontology. Its primitive modules may remain unlabeled. Reuse is established behaviorally and causally: the same parameters must operate across domains, accept compatible arguments from different pathways, support unseen compositions, and be specifically necessary for the relevant behavior.

---

## 2. Central research question

Given several domains with partially shared causal structure, can a network discover or use a sparse computational basis that:

1. reuses computation when the domains genuinely share structure;
2. transfers that computation to withheld domain-operation combinations;
3. preserves domain-specific qualifiers and exceptions;
4. avoids transferring rules where the analogy stops;
5. reduces duplicated parameters or active computation; and
6. eventually produces real hardware savings?

The proposal is not that a concept such as containment must occupy one vector or one named module. A visible relation may decompose into several reusable operations, and different domains may share only a subset of them.

For example, containment might involve:

```text
bind asymmetric roles
→ evaluate membership or enclosure
→ apply a boundary or equality convention
→ derive consequences
```

Geometry, sets, hierarchies, and programs may share some of these operations while differing in others.

---

## 3. What the model is and is not claiming

### 3.1 Intended claim

The intended claim is functional:

> A sparse, reusable computational basis can preserve capability, improve transfer to new compositions, and potentially reduce duplicated parameters and active computation.

### 3.2 Claims not required

The experiment does **not** require proving that:

- a latent variable philosophically “is” containment;
- human semantic categories are the network's natural ontology;
- every relation corresponds to exactly one module;
- the internal representation is unique or identifiable;
- natural language can already be decomposed reliably;
- theoretical sparse FLOPs automatically imply lower latency or energy use.

Neural representations can be transformed without changing their function. Therefore, the target is a **functional equivalence class**: a mechanism counts as reusable if it supports the expected interventions, transfers across domains, binds to new arguments, composes with other mechanisms, and is causally necessary for the associated behavior.

### 3.3 Why oracle information is allowed initially

The oracle isolates the core architectural question from harder upstream questions. If procedural reuse fails even with correct bindings and procedure information, learned parsing and routing will not rescue the central mechanism. If it succeeds, later experiments can remove oracle support one component at a time.

Oracle success is therefore evidence that reuse is possible under ideal decomposition, not evidence that the system can autonomously discover that decomposition.

---

## 4. Hypotheses

The research program separates four hypotheses that must not be bundled into one success claim.

### H1: Compositional advantage

Reusable primitives improve performance on valid, withheld domain-operation combinations or unseen procedural compositions relative to suitable non-sharing and generic baselines.

### H2: Theoretical resource advantage

At matched quality, the sparse primitive architecture uses meaningfully fewer active parameters, active FLOPs, or stored duplicate parameters than an appropriate dense or non-sharing alternative.

### H3: Real hardware advantage

The theoretical savings produce measurable improvements in wall-clock latency, throughput, training time, memory, or energy on specified target hardware.

H3 has two interpretations:

- **H3a — prototype advantage:** the small experimental implementation is measurably faster or more efficient;
- **H3b — crossover evidence:** measured compute, dispatch, launch, and memory costs support a credible estimate of the scale at which sparse execution becomes advantageous.

A tiny model may fail H3a because routing overhead dominates without refuting the possibility of H3b.

### H4: Correct boundary of reuse

The architecture transfers rules that are genuinely shared without catastrophically transferring rules that are domain-specific.

H4 is distinct from generalization. A model can show positive transfer while also producing unacceptable negative transfer.

---

## 5. Operational definition of a reusable primitive

A module is not reusable merely because a router selects it in multiple domains. The same module could implement unrelated functions conditional on its input.

A candidate primitive must satisfy a consistent pattern across these tests:

### 5.1 Parameter recurrence

The same parameters execute for compatible tasks originating in multiple domains.

### 5.2 Cross-domain interchangeability

An intermediate representation produced by one domain adapter can be processed through the same primitive and consumed by another compatible pathway without destroying the relevant behavior.

### 5.3 Counterfactual consistency

Changing irrelevant domain or surface details while holding the operative structure constant leaves the primitive's central result stable.

### 5.4 Argument rebinding

The primitive continues to function when applied to new entities occupying the same typed roles.

### 5.5 Composability

The primitive can participate in a new valid procedure with other primitives rather than succeeding only in memorized full-task configurations.

### 5.6 Causal specificity

Disabling, corrupting, or replacing the primitive damages the associated behavior across relevant domains while leaving unrelated capabilities substantially intact.

No single probe or latent-space visualization is sufficient. The evidence must come from behavioral interchange and causal intervention.

---

## 6. Formal task world

Version one uses a synthetic environment with a known typed relational algebra. Natural language is deliberately excluded.

### 6.1 Latent world

Each task is generated from a latent world:

\[
G = (V, E, R, Q)
\]

where:

- \(V\) is a set of typed entities;
- \(E\) is a set of relational facts or edges;
- \(R\) is the set of formally specified relation types;
- \(Q\) contains qualifiers and domain conventions.

Every relation must have an explicit contract defining:

- argument types and arity;
- ordered argument roles;
- symmetry or asymmetry;
- directionality;
- reflexivity;
- transitivity;
- valid composition rules;
- qualifier semantics;
- and the exact interpretation of each query.

This avoids confusing dataset ambiguity with reasoning failure. For example, adjacency must be distinguished from reachability, membership from subset containment, and a transitive precedence relation from an arbitrary directed edge.

### 6.2 Initial relations

Version one should use four relations selected from a formally coherent set such as:

- equality;
- membership;
- containment or subset relation;
- precedence;
- overlap;
- connectivity or reachability.

The final four must be chosen when the data-generator contract is written. Relations and actions with incompatible input-output signatures should not be mixed in the first experiment merely to increase variety.

### 6.3 Rendered domains

The same or partially shared latent structure is rendered into three controlled domains:

\[
x_d = \rho_d(G)
\]

Candidate domains include:

- set-like structures;
- object hierarchies;
- simplified two-dimensional geometry;
- simple formal programs.

Each renderer \(\rho_d\) changes the surface representation while preserving the formally declared semantics for the exact-isomorphism regime.

### 6.4 Renderer protections

To prevent the model from merely memorizing one renderer inversion, the data should include:

- multiple renderers per domain;
- randomized vocabularies and symbol assignments;
- altered surface ordering;
- irrelevant distractors;
- held-out renderers;
- larger held-out graph structures;
- and overlapping surface statistics for tasks with different relations.

Renderer generalization, structural generalization, and unseen domain-operation recombination must be evaluated separately.

---

## 7. Two semantic regimes

### 7.1 Exact-isomorphism regime

In the first regime, the selected domains genuinely share the relevant latent operation. This regime asks only:

> Can the architecture reuse computation when reusable structure truly exists?

Success must not be presented as evidence that naturally occurring domains are perfectly isomorphic.

### 7.2 Partial-analogy regime

After exact reuse is demonstrated, controlled domain-specific deviations are introduced. Examples include:

- inclusive versus exclusive geometric boundaries;
- directed versus undirected connectivity;
- discrete set membership versus approximate spatial enclosure;
- transitive hierarchy containment versus non-transitive program behavior;
- equality-by-value versus equality-by-identity.

Exceptions must be explicitly generated, balanced, and protected from superficial frequency cues. This regime tests whether the system learns the boundary of reuse.

### 7.3 Positive and negative transfer

Report valid and invalid transfer separately.

One operational definition is:

\[
T^+ = Q_{\text{shared}}^{\text{valid}}
      - Q_{\text{separate}}^{\text{valid}}
\]

\[
T^- = Q_{\text{separate}}^{\text{exceptions}}
      - Q_{\text{shared}}^{\text{exceptions}}
\]

where the separate model is a matched non-sharing procedural executor. High \(T^+\) is desirable; high \(T^-\) is harmful. Raw scores must be reported alongside these differences.

---

## 8. Version-one model architecture

The initial architecture should be intentionally small and constrained.

### 8.1 Proposed scale

- four-layer transformer encoder or comparably small backbone;
- hidden dimension approximately 128–256;
- eight candidate primitive modules;
- two role-preserving argument slots;
- explicit context and qualifier inputs;
- at most two sequential primitive calls;
- fixed maximum execution depth;
- oracle argument bindings initially;
- oracle or learned abstract procedure selection, depending on stage;
- no unrestricted residual path in the exact-isomorphism experiment.

These values are starting points, not scientific constants. Changes must be recorded rather than silently introduced.

### 8.2 Structured inputs

The model receives a structured task record conceptually resembling:

```text
task representation:
  world or relevant facts
  argument slots:
    role_1 → entity_1
    role_2 → entity_2
  context/domain
  qualifiers
  oracle procedure information, if enabled for this stage
```

Argument roles must be preserved. A bag containing two entities is insufficient for asymmetric relations.

### 8.3 Primitive modules

The eight primitive modules should initially have identical architecture and independent initialization. They should not be permanently labeled `membership`, `containment`, or other dataset relations.

The dataset's relations describe the task. The network's primitives describe its computation. The two need not align one-to-one.

A mature procedure may be a sparse ordered composition:

\[
p = C_{k_2} \circ C_{k_1},
\qquad k_1,k_2 \in \{1,\ldots,K\}
\]

rather than an unordered mixture. Version one should use a fixed maximum of two sequential calls so that execution remains measurable and interpretable.

### 8.4 Primitive call contract

Before implementation, the interface of one primitive call must be fixed. A possible form is:

\[
C_k(s_t, e_t, q_t) \rightarrow (s_{t+1}, v_t)
\]

where:

- \(s_t\) is compact execution state;
- \(e_t\) is one selected fact, entity pair, or typed input object;
- \(q_t\) is context or qualifier information;
- \(s_{t+1}\) is updated state;
- \(v_t\) is optional output or confidence information.

The exact contract must restrict what each call can observe. If a single module sees the entire problem and has sufficient capacity to solve it, nominal two-step execution provides no evidence of learned composition.

### 8.5 No residual in the first regime

The exact-isomorphism experiment starts without an unrestricted residual channel because the oracle inputs should expose all necessary task information. This prevents the network from bypassing the primitive mechanism.

A narrow, controlled residual is introduced only with the partial-analogy regime. Its capacity should be varied experimentally and constrained through dimensionality, dropout, noise, bandwidth penalties, or residual swapping.

### 8.6 Later compatibility mechanism

When learned routing is introduced, a primitive may expose a learned interface signature \(q_k\), compared with a request representation through a cheap shared compatibility function:

\[
a_k = \phi(z_{\text{request}},q_k)
\]

The route can remain cached while compatibility is high. Rerouting should be an exception rather than a full decision repeated at every token and layer.

This mechanism is outside the minimum oracle experiment but should remain compatible with its module interface.

---

## 9. Meaning and limits of the oracle

### 9.1 Oracle-provided information

Depending on the stage, the oracle may provide:

- correct typed argument bindings;
- the abstract operations required by the task;
- a procedure skeleton;
- or the order in which abstract operations should occur.

### 9.2 Information the oracle must not provide

The oracle must not provide:

- the target answer;
- a code uniquely correlated with the target answer;
- a route token that makes the world facts unnecessary;
- hidden information unavailable from the task definition;
- or a procedure description so specific that execution becomes a lookup table.

### 9.3 Leakage tests

Every oracle channel must be audited through:

- a procedure-token-only probe;
- argument-only and context-only probes;
- shuffled-fact evaluation with procedure tokens held fixed;
- replaced-argument evaluation;
- label-balance checks within each procedure pattern;
- and mutual-information or predictive-leakage estimates where useful.

If the output can be predicted well without processing the relevant world state, the oracle representation is invalid.

### 9.4 Mapping oracle operations to unlabeled modules

Oracle selection appears to conflict with unlabeled primitive modules. The resolution differs by stage.

#### Stage A: fixed arbitrary permutation

For Stage A, the mapping is a one-to-one permutation sampled deterministically from the experimental seed:

\[
\pi_s: O \rightarrow \{C_1,\ldots,C_K\}
\]

where \(O\) is the Stage A abstract-operation vocabulary and \(s\) is the run seed. The permutation is fixed before training and never optimized. The oracle supplies an abstract operation token; \(\pi_s\) selects exactly one primitive module for that step.

Consequences:

- module identities remain semantically arbitrary across seeds;
- Stage A contains no learned routing component;
- one abstract operation maps to exactly one module within a run;
- an operation does not select multiple modules in Stage A;
- unused candidate modules remain unused unless the frozen implementation specification assigns every operation token;
- module interpretations must be aligned across seeds by behavior or permutation matching, never by raw module index.

The implementation specification must choose \(|O|\) relative to \(K\) and state whether unused modules exist. It may not change the one-to-one, fixed-per-seed rule for Stage A without recording a protocol amendment.

#### Stage B and later: learned mapping

Beginning in Stage B, the oracle may still supply the abstract operation token while a learned mapping selects a module:

\[
o_i \xrightarrow{\pi_\theta} C_k
\]

The oracle specifies the required abstract step; the learned mapping \(\pi_\theta\) determines which arbitrary module realizes it. The exact parameterization, sparsification, optimization, and regularization of \(\pi_\theta\) must be frozen in the implementation specification. Module identities remain permutation-equivalent and need not carry human-readable names.

Alternative experiments may temporarily assign semantic module identities or pretrain atomic modules, but they are separate, additional-supervision ablations and must not be reported as canonical Stage A.

---

## 10. Oracle stages

Oracle support is removed progressively so failures can be attributed to the correct component.

| Stage | Bindings | Abstract selection | Order | Main question |
|---|---|---|---|---|
| A | Oracle | Oracle through fixed per-seed permutation | Oracle | Can the modules execute the supplied procedure without learned routing? |
| B | Oracle | Learned | Oracle | Can the model select reusable modules? |
| C | Oracle | Learned | Learned | Can it construct a bounded two-step procedure? |
| D | Learned | Learned | Learned | Can it recover the formal decomposition? |
| E | Learned from surface forms | Learned | Learned | Can it survive ambiguity and language-like variation? |

Stage A is a capacity and mechanism test, not a discovery test. Stage E should not begin until the earlier stages succeed.

Additional sub-stages may distinguish:

- oracle operator and oracle bindings;
- oracle bindings with learned primitive composition;
- oracle primitive selection with learned order;
- and fully learned selection and composition.

---

## 11. Task classes and data split

### 11.1 Direct tasks

Direct tasks require one formally specified operation, such as testing equality, membership, or a directly stored relation.

### 11.2 Compositional tasks

Compositional tasks require a valid, explicitly declared inference involving two steps. Examples must respect the typed relational algebra. They must not assume transitivity, symmetry, or reachability unless those properties are part of the relation contract.

Some tasks should require one call and others two. Unnecessary calls should carry an execution cost.

### 11.3 Withheld combinations

For each relation or procedure family:

- train it in a subset of domains;
- withhold at least one domain-operation combination;
- ensure the withheld domain's adapter is trained through other operations;
- rotate held-out combinations across relations and domains.

Evaluation must distinguish:

1. a new combination of known operation and known domain;
2. a new renderer for known semantics;
3. a larger or topologically novel latent structure;
4. a new valid primitive composition;
5. a controlled domain-specific exception.

---

## 12. Training objectives

Ordinary task performance remains necessary. Additional objectives should force functional reuse rather than merely similar embeddings.

A general form is:

\[
L = L_{\text{task}}
  + \lambda_1 L_{\text{interchange}}
  + \lambda_2 L_{\text{counterfactual}}
  + \lambda_3 L_{\text{composition}}
  + \lambda_4 L_{\text{compute}}
  + \lambda_5 L_{\text{information}}
\]

where:

- \(L_{\text{task}}\) measures answer correctness;
- \(L_{\text{interchange}}\) requires compatible cross-domain latent substitutions to preserve behavior;
- \(L_{\text{counterfactual}}\) requires changes to the correct factor to have the correct behavioral consequence;
- \(L_{\text{composition}}\) rewards success on unseen valid combinations;
- \(L_{\text{compute}}\) penalizes the executed graph, including primitive FLOPs and routing overhead;
- \(L_{\text{information}}\) constrains shortcut channels without discarding necessary distinctions.

The compute objective should measure the whole executed graph rather than merely counting selected modules:

\[
L_{\text{compute}} \propto
\sum_{k,t} g_{k,t}\,\operatorname{FLOPs}(C_k)
+ \operatorname{routing\ overhead}
\]

Compute or sparsity penalties should be introduced gradually. Applying them too early can produce a cheap but incapable model.

Loss weights and schedules must be recorded. They should not be adjusted independently for the proposed model without giving baselines comparable tuning opportunity.

---

## 13. Required controls and baselines

No single “equal MoE” baseline is sufficient because oracle information, module interfaces, sharing, sparsity, and sequential execution are separate variables.

### 13.1 Core baselines

1. **Dense transformer:** ordinary non-modular baseline.
2. **Parallel learned-routing MoE:** conventional top-\(k\) expert baseline.
3. **Oracle-informed parallel MoE:** receives the same allowable oracle information as the OPM.
4. **Untied procedural executor:** same typed interfaces, oracle information, and sequential execution as the OPM, but separate modules per domain.
5. **Dense-execution shared-primitive model:** same shared modules but executes all candidates before applying the same selection mask.
6. **Oracle Primitive Model:** shared primitives with oracle decomposition and sparse bounded execution.
7. **Learned Primitive Model:** progressively replaces oracle components with learned components.
8. **Explicit oracle executor:** symbolic or directly programmed upper bound for the formal task.

### 13.2 Most informative direct pair

The cleanest comparison is:

| Feature | Untied procedural executor | Oracle Primitive Model |
|---|---:|---:|
| Oracle bindings | Same | Same |
| Procedure information | Same | Same |
| Module interface | Same | Same |
| Sequential depth | Same | Same |
| Training data | Same | Same |
| Target active FLOPs | Matched | Matched |
| Domain-level parameter sharing | No | Yes |

This pair tests domain-specific procedural modules versus weight-tied procedural modules.

### 13.3 Controls for parameter tying and data pooling

Sharing changes both parameter organization and the diversity of experience received by each module. Therefore include, when feasible:

- **Untied clones:** domain modules begin identically but may diverge during training.
- **Periodic synchronization:** domain modules train separately but are periodically weight-averaged.
- **Continuous weight tying:** all domains update and execute the same parameters.

This ladder distinguishes shared initialization, cross-domain consolidation, and continuous reusable computation:

```text
same initialization
→ untied training
→ periodic synchronization
→ continuous weight tying
```

Update counts may be resampled for comparison, but equal update counts do not make the information content identical. That limitation must be acknowledged.

### 13.4 Dense-execution control

To isolate sparsity without adding a different aggregation mechanism, the dense shared-primitive model should execute all modules but apply the same selected transition at the output:

\[
u_k=C_k(x)\quad\forall k,
\qquad y=u_{k^\star}
\]

For two sequential steps, all candidates execute at each step before the selected transition is retained.

### 13.5 Capacity-advantaged generic controls

At a later phase, include a conventional MoE with deliberately greater capacity or active computation. Prefer a small scaling curve over one arbitrarily selected “stronger MoE.” The question is whether the primitive model moves the quality-compute Pareto frontier, not whether it defeats one convenient baseline.

---

## 14. Matching and fairness

No comparison can generally match all of the following simultaneously:

- total parameters;
- active parameters;
- inference FLOPs;
- training FLOPs;
- memory;
- wall-clock time;
- and achieved quality.

Therefore, report several comparisons.

### 14.1 Equal total parameters

Hold stored capacity approximately constant; compare quality and active cost.

### 14.2 Equal active FLOPs

Hold theoretical inference compute approximately constant; compare quality and total capacity.

### 14.3 Equal training compute

Hold total training FLOPs constant; compare final quality and inference cost.

### 14.4 Quality matched

Train or scale models until they reach a declared quality threshold; compare training cost to threshold, inference cost, total parameters, and wall-clock behavior.

### 14.5 Scaling curves

Where resources permit, fit small empirical frontiers rather than relying on a single matched model:

\[
Q=f(\text{active FLOPs},\text{total parameters},\text{training compute})
\]

Hyperparameter-search budgets and stopping rules must also be comparable.

---

## 15. Causal and anti-shortcut tests

The following tests are central evidence, not optional interpretability extras.

### 15.1 Primitive ablation

Disable a candidate shared primitive and measure whether the associated behavior degrades across relevant domains.

### 15.2 Specificity test

Verify that unrelated relations and tasks remain substantially intact after ablation.

### 15.3 Primitive replacement

Replace the selected primitive with noise, a different primitive, or a frozen reference module. Confirm predictable behavioral changes.

### 15.4 Cross-domain interchange

Insert a compatible latent or execution state originating in one domain into another domain's pathway. Test whether the operative result is preserved.

### 15.5 Adapter-only test

Disable the shared primitive and test whether adapters or domain-specific pathways can still solve the task. If they can, duplication has migrated outside the shared mechanism.

### 15.6 Residual-only test

When residuals are introduced, determine whether the residual alone predicts the task answer or operative relation.

### 15.7 Residual swap

Swap residuals between semantically matched examples. Core relational behavior should remain stable while legitimate contextual details may change.

### 15.8 Representation leakage probes

Test whether:

- operator or procedure state encodes the target answer;
- argument representations leak domain unnecessarily;
- context encodes entity identity;
- routing decisions memorize surface vocabulary;
- or oracle tokens bypass task processing.

Probe results are diagnostic and must be supported by interventions before drawing causal conclusions.

### 15.9 Surface-correlation reversal

Reverse correlations such as particular shapes, symbols, or vocabularies being associated with particular relations. Performance should depend on structure rather than surface cues.

---

## 16. Metrics

### 16.1 Capability and transfer

- direct-task accuracy;
- compositional-task accuracy;
- withheld domain-operation accuracy;
- held-out renderer accuracy;
- larger-structure accuracy;
- valid-transfer gain \(T^+\);
- exception-transfer harm \(T^-\);
- calibration or confidence where meaningful.

### 16.2 Causal reuse

- cross-domain interchange success;
- relevant performance loss under primitive ablation;
- unrelated performance preservation under ablation;
- adapter-only and residual-only performance;
- route stability across surface transformations;
- primitive usage consistency across domains.

### 16.3 Theoretical resources

- total parameters;
- active parameters;
- forward FLOPs;
- backward FLOPs;
- optimizer-state memory;
- activation memory;
- steps to declared quality threshold;
- total training FLOPs to threshold.

### 16.4 Hardware behavior

- wall-clock training time;
- single-example latency;
- large-batch throughput;
- peak VRAM;
- average accelerator utilization;
- routing and dispatch time;
- kernel-launch overhead;
- module-loading and communication costs;
- energy, if reliably measurable.

All hardware comparisons must use the same specified hardware and software environment.

---

## 17. Statistical and experimental protocol

Before the primary runs:

- predefine train, validation, and test generators;
- freeze held-out operator-domain combinations;
- freeze exception sets and held-out renderers;
- define success thresholds;
- define early-stopping and compute budgets;
- specify model-matching rules;
- specify the number of random seeds;
- and record the allowed hyperparameter-search budget.

All models should be evaluated on the same latent worlds, renderer seeds, exception cases, and withheld combinations. Report paired differences per world where possible rather than comparing independently sampled aggregate scores.

Confidence intervals, variation across seeds, and both raw scores and derived transfer measures must be reported.

---

## 18. Proposed experimental phases

The full factorial matrix should not be run before the core mechanism is established.

### Phase 1: Reuse under oracle decomposition

Models:

- untied procedural executor;
- weight-tied Oracle Primitive Model;
- untied-clone control;
- optional periodic-synchronization control;
- explicit oracle executor upper bound.

Questions:

- Can the modules execute the supplied procedures?
- Does continuous sharing improve valid held-out transfer?
- Does sharing remain causally localized?

Primary hypotheses: H1 and idealized H4.

### Phase 2: Sparse execution

Models:

- dense-execution weight-tied primitive model;
- sparse-execution weight-tied primitive model.

Questions:

- Does sparse execution preserve capability?
- Does it reduce active theoretical resources?
- What overhead does it introduce?

Primary hypothesis: H2.

### Phase 3: Learned decomposition

Models:

- progressively learned-routing and learned-binding primitive models;
- vanilla parallel MoE;
- oracle-informed parallel MoE;
- relevant ablations without interchange or counterfactual training.

Questions:

- Can the model select and compose primitives without oracle selection?
- Can it learn bindings without corrupting reuse?
- Is procedural structure better than generic conditional capacity?

Primary hypotheses: autonomous H1 and H4.

### Phase 4: Practical frontier

Models:

- dense transformer scaling curve;
- parallel MoE scaling curve;
- procedural model scaling curve;
- capacity-advantaged generic controls.

Questions:

- Does the model improve the quality-compute frontier?
- Do theoretical savings become wall-clock or energy savings?
- Is there credible evidence of a hardware crossover scale?

Primary hypothesis: H3.

### Phase 5: Increasingly realistic inputs

Only after earlier phases succeed:

- learn argument bindings from structured surfaces;
- introduce ambiguity and multiple candidate bindings;
- introduce language-like renderers;
- then consider natural language, code, diagrams, or multimodal inputs.

The system may retain one primary binding, one alternative, an unresolved-information mask, and a confidence margin rather than allowing uncontrolled branching.

---

## 19. Success criteria

Exact numerical thresholds must be set before primary experimentation. Conceptually, success requires simultaneous evidence in the relevant phase.

### H1 succeeds if

- the shared primitive model shows statistically reliable improvement on valid withheld combinations relative to an appropriate matched non-sharing control;
- and the result survives renderer changes and surface-correlation reversals.

### H2 succeeds if

- matched quality is reached with meaningfully fewer active FLOPs, active parameters, or duplicated stored parameters;
- and the calculation includes the complete executed graph and routing overhead.

### H3 succeeds if

- wall-clock, throughput, memory, or energy improves on specified hardware;
- or component measurements support a clearly labeled and credible crossover estimate.

### H4 succeeds if

- valid transfer improves without a substantial rise in controlled exception errors;
- and interventions show domain-specific differences remain available through the intended pathway rather than duplicated full capabilities.

### Evidence of genuine causal reuse additionally requires

- successful cross-domain interchange;
- relevant cross-domain degradation under shared-module ablation;
- preservation of unrelated abilities under that ablation;
- inability of adapters or residuals to reproduce the full capability alone;
- and resistance to oracle and surface shortcuts.

---

## 20. Falsification and informative failure

The hypothesis is weakened if:

- the shared primitive model cannot outperform a matched non-sharing procedural executor on valid withheld combinations;
- apparent gains disappear under renderer changes or reversed correlations;
- adapters or residuals repeatedly relearn complete capabilities;
- primitive ablations do not produce cross-domain, behavior-specific effects;
- exact-isomorphism transfer works but controlled exceptions cause unacceptable negative transfer;
- learned routing cannot recover the oracle model's behavior;
- a conventional MoE matches the quality-compute frontier without procedural structure;
- causal reuse requires so much regularization that ordinary task competence collapses;
- or routing, sequential fragmentation, and memory movement erase theoretical compute savings.

Different failures implicate different claims:

- Reuse works but sparse execution is slow: H1 may succeed while H3 fails.
- Dense modular execution works but sparse selection fails: routing or systems execution is the defect.
- Exact transfer works but exceptions fail: reuse exists, but H4 fails.
- Oracle execution works but learned selection fails: the mechanism is viable, but autonomous decomposition remains unsolved.
- Generic oracle-informed MoE matches OPM: privileged decomposition or structured supervision, rather than primitive organization, explains the benefit.
- Untied and tied procedural models perform alike: enforced sharing did not provide measurable advantage in the tested regime.

Negative results remain informative. They may indicate that useful neural redundancy is not organized into clean reusable procedural units, that the proposed interfaces are wrong, or that dense transformers already exploit sharing more effectively than expected.

---

## 21. Design invariants

The following principles should persist unless explicitly revised with a recorded rationale:

1. **Behavior over labels.** Primitive names and latent visualizations do not establish reuse.
2. **Oracle honesty.** Oracle information may decompose a task but may not encode its answer.
3. **Role preservation.** Ordered and typed argument roles must remain distinguishable.
4. **Bounded execution.** Primitive count, depth, routing revisions, and executed FLOPs must be constrained and measured.
5. **No hidden duplicate model.** Adapters and residuals must be causally tested for bypass behavior.
6. **Sharing has a boundary.** Positive and negative transfer must be measured separately.
7. **Controls receive equivalent advantages.** Information, interfaces, training data, and tuning must be matched or isolated.
8. **Theory and hardware are separate.** Active parameter count is not latency, energy, or training cost.
9. **Oracle success is an upper bound.** It is not evidence of autonomous decomposition.
10. **Add complexity only after mechanism success.** Parsing, language ambiguity, deep composition, and dynamic routing enter progressively.

---

## 22. Scientific-core implementation gate

The OPM v1 scientific core is **not implementation-ready** until every item below is frozen in `OPM_V1_IMPLEMENTATION_SPEC.md`, validated for internal consistency, and assigned a stable specification version.

### 22.1 Required specification artifacts

The companion implementation specification must contain all of the following. A heading without executable detail does not satisfy the requirement.

#### A. Typed relational algebra

For every entity and relation:

- canonical identifier;
- entity types;
- argument arity and ordered role names;
- valid and invalid type signatures;
- symmetry, directionality, reflexivity, and transitivity flags;
- complete one-step and two-step inference tables;
- qualifier interaction tables;
- query grammar;
- target-label derivation algorithm;
- invalid-world constraints;
- and worked positive and negative examples.

The algebra must make it mechanically possible to generate a world, verify its consistency, enumerate valid queries, and compute each target without learned code.

#### B. Generator and renderer algorithms

For each domain and renderer:

- exact entity vocabulary generation;
- exact fact serialization;
- query serialization;
- ordering and permutation rules;
- distractor-number and distractor-type distributions;
- graph-size and topology distributions;
- qualifier distributions;
- randomized-symbol rules;
- held-out renderer transformations;
- rejection-sampling conditions;
- class-balancing procedure;
- and pseudocode precise enough to translate directly into deterministic code.

The specification must state which latent information is shared across domains and which information is intentionally domain-specific.

#### C. Dataset manifest

- exact train, validation, interpolation-test, recombination-test, renderer-test, structural-test, and exception-test sizes;
- generator seeds or immutable seed ranges for every split;
- held-out domain-operation matrix;
- held-out composition matrix;
- overlap and contamination checks;
- target-class balance tolerances;
- duplicate policy;
- and immutable dataset fingerprint procedure.

No primary test seed may be used for model selection or loss-weight tuning.

#### D. Tensor and module contracts

For every tensor crossing a component boundary:

- symbolic name;
- shape including batch and sequence axes;
- dtype;
- allowable value range;
- mask semantics;
- initialization source;
- gradient-flow rule;
- and semantic contract.

This includes world encodings, argument slots, context, qualifiers, oracle tokens, execution state, selected evidence, primitive outputs, stop decisions, and answer logits.

#### E. Primitive computation boundary

The specification must define exactly:

- what one primitive call can observe;
- whether it receives one fact, a selected fact pair, or an aggregated memory read;
- how evidence is selected;
- which information persists in execution state;
- whether a primitive can access the original world representation after step zero;
- primitive depth, width, activation, normalization, residual connections, and parameter count;
- state initialization and state-transition equations;
- final decoder inputs;
- and why one call cannot trivially solve every two-step task.

This section defines what “one computational step” means. It is part of the scientific hypothesis, not an implementation convenience.

#### F. Oracle channel contract

- complete oracle-token vocabulary;
- token-to-task meaning;
- number and order of tokens per task class;
- role-binding representation;
- fact-selection information, explicitly present or explicitly forbidden;
- information visible at each execution step;
- Stage A fixed-per-seed permutation algorithm;
- structural prevention of target leakage;
- and the exact probes and acceptance thresholds used to audit leakage.

The specification must demonstrate that target labels remain balanced or unpredictable conditional on oracle tokens alone.

#### G. Exact model configurations

For the proposed model and every baseline:

- backbone depth and width;
- attention configuration;
- embedding sizes;
- primitive or expert count, depth, and width;
- active-module count;
- router architecture where applicable;
- sequential depth;
- decoder architecture;
- total and active parameter calculation;
- theoretical forward and backward FLOP formulas;
- and a machine-readable configuration block.

Every claimed match must include a numerical accounting table and its tolerance.

#### H. Operational loss definitions

For every enabled loss:

- exact equation;
- exact examples or pairs to which it applies;
- reduction rule;
- negative-sampling rule;
- coefficient;
- activation schedule;
- gradient destinations and stopped-gradient paths;
- numerical-stability handling;
- and ablation condition.

Interchange, counterfactual, composition, compute, and information losses may not remain conceptual names in the implementation specification.

#### I. Training protocol

- optimizer and all parameters;
- initialization scheme;
- learning-rate and regularization schedules;
- batch construction and sampling proportions;
- number of examples or tokens per update;
- maximum steps and compute budget;
- checkpoint cadence;
- validation cadence;
- early-stopping rule;
- gradient clipping and precision policy;
- deterministic-mode expectations;
- allowed hyperparameter search space;
- search budget per model family;
- and model-selection rule.

#### J. Intervention and metric algorithms

For every causal test and metric:

- intervention location and exact replacement value;
- construction of matched control examples;
- aggregation formula;
- required comparison model;
- uncertainty estimate;
- and interpretation limits.

This includes primitive ablation, replacement, interchange, adapter-only evaluation, residual-only evaluation, residual swaps, leakage probes, surface-correlation reversal, \(T^+\), and \(T^-\).

#### K. Preregistered statistical protocol

- experimental run seeds;
- paired-evaluation keys;
- primary and secondary endpoints;
- statistical tests;
- confidence-interval method;
- multiple-comparison correction where applicable;
- minimum effect sizes;
- success, failure, and inconclusive thresholds for H1–H4;
- hardware-repeat count;
- and rules for excluding failed or corrupted runs.

#### L. Environment and reproducibility manifest

- target hardware;
- framework and library versions;
- deterministic-kernel settings;
- precision mode;
- compilation settings;
- profiler configuration;
- artifact directory structure;
- configuration and code revision identifiers;
- and the command sequence for regenerating data, training models, and reproducing tables.

### 22.2 Entry criteria

Scientific-core coding may begin only when:

1. all artifacts A–L exist;
2. every normative field is either assigned a value or explicitly marked out of scope for v1;
3. equations, pseudocode, and machine-readable configuration agree;
4. a dry-run trace can follow one direct task and one compositional task from latent-world generation through target derivation and model execution;
5. parameter and FLOP accounting closes numerically for every Phase 1 model;
6. oracle-only leakage tests are defined before any primary training run;
7. held-out test manifests are frozen and fingerprinted;
8. and the specification has a version identifier and approval record.

### 22.3 Exit criteria

An implementation conforms to OPM v1 only if:

- all primary scientific choices originate in the approved implementation specification;
- the run records the specification, configuration, dataset, and code revisions;
- deviations are declared before examining primary test results or are labeled exploratory afterward;
- all required controls for the claimed hypothesis are present;
- and results are reported against the preregistered endpoints rather than a post hoc substitute.

### 22.4 No-invention rule

If an implementer encounters ambiguity that can change the meaning, difficulty, information content, compute accounting, or fairness of the experiment, the implementer must stop that scientific-core path and record a specification issue. The implementer may not silently choose a plausible interpretation.

Routine engineering choices that cannot affect those properties may be made locally, but they must still be recorded when they affect reproducibility.

### 22.5 Unresolved scientific decisions

The following decisions remain open at the research-charter level and are blocking inputs to the companion specification:

1. The exact four relations and their typed algebra.
2. The exact three domains and renderer contracts.
3. The input and output contract of one primitive call.
4. What constitutes one computational step.
5. The oracle token vocabulary and leakage protections.
6. The Stage B-and-later learned mapping from abstract tokens to modules. Stage A is already fixed as a deterministic per-seed one-to-one permutation.
7. The exact training split and rotated held-out combinations.
8. Model widths, module widths, parameter-matching rules, and compute accounting method.
9. Training objectives, schedules, and tuning budgets.
10. Random-seed count, statistical tests, and numerical success thresholds.
11. Target hardware and measurement methodology for H3.
12. The controlled exception families introduced for H4.

These choices must be resolved explicitly in `OPM_V1_IMPLEMENTATION_SPEC.md`. They must not drift into source code, undocumented configuration defaults, or autonomous-agent judgment.

## 23. Conformance, change control, and traceability

### 23.1 Normative language

In this document and the companion implementation specification:

- **must** and **must not** indicate requirements for a conforming canonical experiment;
- **should** indicates the preferred choice, from which a deviation requires written rationale;
- **may** indicates a permitted option;
- **exploratory** identifies work that may inform later versions but cannot replace a preregistered canonical result.

### 23.2 Requirement identifiers

The companion implementation specification must assign stable identifiers to its normative requirements, using categories such as:

```text
ALG-*   relational algebra
REN-*   renderers and generators
DAT-*   dataset and splits
MOD-*   tensors and model components
ORC-*   oracle channel
LOS-*   losses
TRN-*   training protocol
BAS-*   controls and matching
MET-*   metrics and interventions
STA-*   statistics and thresholds
SYS-*   hardware and reproducibility
```

Tests, configurations, and result tables should cite these identifiers so that compliance is auditable.

### 23.3 Decision log

Every scientific decision must record:

```text
decision_id:
date:
status: proposed | approved | superseded
question:
selected_option:
alternatives_considered:
rationale:
affected_requirements:
approved_by:
```

### 23.4 Amendments

Any change after preregistration must include:

- the exact previous and new requirement;
- the reason for the change;
- whether primary test results had been observed;
- affected datasets, models, and claims;
- and whether previous runs remain comparable.

Changes made after viewing primary results produce a new exploratory or confirmatory protocol version; they do not silently modify the original protocol.

### 23.5 Traceability matrix

Before primary runs, maintain a table linking:

```text
research hypothesis
→ implementation requirements
→ configuration fields
→ automated tests
→ produced artifacts
→ reported metrics
→ acceptance decision
```

This ensures that no hypothesis is declared supported by measurements that do not actually test it.

### 23.6 Source-of-truth precedence

When two artifacts disagree, use this order:

1. an approved protocol amendment applying to the relevant version;
2. the approved `OPM_V1_IMPLEMENTATION_SPEC.md` for implementation details;
3. this charter for research meaning, invariants, required controls, and claim boundaries;
4. frozen machine-readable experiment configuration;
5. source-code defaults;
6. comments, notebooks, issue discussions, and informal descriptions.

No lower-priority artifact may silently override a higher-priority artifact. A conflict between the charter and implementation specification must be resolved explicitly: the implementation specification cannot waive a charter invariant merely by contradicting it.

### 23.7 Specification lifecycle

The project advances through the following states:

```text
CHARTER_ONLY
  → SPEC_DRAFTING
  → SPEC_REVIEW
  → SPEC_APPROVED
  → IMPLEMENTATION_VALIDATION
  → PILOT_ONLY
  → PROTOCOL_FROZEN
  → PRIMARY_RUNS
  → ANALYSIS_LOCKED
  → REPORTED
```

State meanings:

- **CHARTER_ONLY:** only this charter is authoritative; scientific-core work is blocked.
- **SPEC_DRAFTING:** artifacts A–L are being resolved; no canonical scientific implementation.
- **SPEC_REVIEW:** the companion specification is complete enough for consistency and leakage review.
- **SPEC_APPROVED:** the companion specification has a version and approval record; scientific-core implementation may begin.
- **IMPLEMENTATION_VALIDATION:** code is checked against requirement identifiers, dry-run traces, and accounting tests.
- **PILOT_ONLY:** training may detect bugs and tune only within the preregistered search plan; primary test manifests remain sealed.
- **PROTOCOL_FROZEN:** code, data manifests, configurations, endpoints, exclusions, and analysis rules are frozen for primary runs.
- **PRIMARY_RUNS:** confirmatory runs execute without protocol changes.
- **ANALYSIS_LOCKED:** the preregistered analysis has executed and outputs are immutable.
- **REPORTED:** results, deviations, artifacts, and claim decisions are published internally or externally.

A lifecycle state change must record the date, responsible party, artifact versions, satisfied entry criteria, and unresolved exceptions. Merely creating a file named `OPM_V1_IMPLEMENTATION_SPEC.md` does not advance the project to `SPEC_APPROVED`.

### 23.8 Approval authority

The companion specification is approved only when the researcher or explicitly designated scientific owner records approval. An autonomous implementation agent may draft, critique, and validate the specification but may not self-approve meaning-bearing scientific choices unless explicitly delegated that authority.

Approval must record:

```text
specification_id:
specification_version:
approved_commit_or_hash:
approved_by:
approval_date:
approved_scope:
known_exceptions:
next_lifecycle_state:
```

### 23.9 Primary-test sealing

Primary test manifests, seeds, and fingerprints must be stored so that routine pilot work does not expose their labels or results. Access sufficient to inspect primary outcomes before protocol freeze converts subsequent work into exploratory analysis unless a predeclared recovery rule applies.

At minimum, the audit record must show:

- when each primary manifest was generated;
- its immutable fingerprint;
- who or what accessed it;
- whether labels or aggregate results were exposed;
- and whether the protocol was already frozen.

### 23.10 Claim ledger

Every reported conclusion must be entered in a claim ledger:

| Claim ID | Hypothesis | Required comparison | Primary metric | Threshold | Result | Status | Deviations |
|---|---|---|---|---|---|---|---|

Allowed statuses are `supported`, `not_supported`, and `inconclusive`. “Not supported” is not automatically evidence of the opposite claim. Exploratory findings must be labeled separately and cannot replace a failed preregistered endpoint.

### 23.11 Canonical next-task directive

While the charter remains in `CHARTER_ONLY` or `SPEC_DRAFTING`, the canonical next scientific task is:

> Produce `OPM_V1_IMPLEMENTATION_SPEC.md` by resolving artifacts A–L. Do not implement the canonical scientific model or dataset. Record every meaning-bearing choice in the decision log, provide executable pseudocode and machine-readable configuration where required, and flag choices requiring researcher approval.

The companion specification may be developed in reviewable increments, but partial completion does not authorize implementation of scientific components whose contracts depend on unresolved sections.

---

## 24. One-paragraph canonical description

The Oracle Primitive Model is a controlled neural architecture experiment in which correct task decomposition is initially supplied so that the viability of cross-domain procedural reuse can be tested independently of parsing and routing. Typed, role-preserving arguments are processed by a small number of sequentially executed, architecturally interchangeable primitive modules whose parameters may be shared across domains. The model is compared with dense, conventional MoE, oracle-informed, untied procedural, and dense-execution controls. Reuse is established through held-out composition, cross-domain interchange, causal ablation, and shortcut-resistant evaluation rather than through module labels or representation similarity. Subsequent stages remove oracle selection and bindings, introduce controlled domain exceptions and constrained residual pathways, and finally measure whether the learned system improves theoretical and real quality-compute frontiers.

---

## 25. Canonical short thesis

> Given multiple domains with partially shared causal structure, test whether a network can use a sparse, reusable computational basis to transfer what is genuinely shared, preserve what is domain-specific, and reduce duplicated computation—first under oracle decomposition, then with the decomposition learned.
