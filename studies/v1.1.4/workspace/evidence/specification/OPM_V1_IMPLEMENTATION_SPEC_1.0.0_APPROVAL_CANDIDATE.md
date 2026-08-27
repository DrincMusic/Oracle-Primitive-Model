# OPM v1 Implementation Specification

## Approved companion to the Oracle Primitive Model charter

```yaml
document_control:
  document_id: OPM-V1-SPEC
  specification_version: 1.0.0
  charter_id: OPM-CHARTER
  charter_version: 1.2.0
  lifecycle_state: SPEC_REVIEW
  approval_status: PENDING_OWNER_CONFIRMATION
  scientific_core_authorized: false
  intended_next_state: SPEC_APPROVED
  normative_keywords: [must, must_not, should, may, exploratory]
```

**Authority:** This document defines the technically reviewed, directly implementable OPM v1 experiment under the constraints of `ORACLE_PRIMITIVE_MODEL.md`. Version 1.0.0 awaits explicit owner approval and does not yet authorize scientific-core implementation or primary experiments.

**Scope:** OPM v1 tests H1 under oracle decomposition. H2 receives theoretical accounting only. H3, H4, natural language, learned bindings, Stage B routing, qualifier-conditioned exceptions, unrestricted residual paths, and large-scale training are out of scope for the canonical v1 run.

**Canonical question:** With identical oracle bindings, procedure tokens, execution interfaces, and comparable active computation, does continuous cross-domain weight tying improve transfer to withheld domain-operation combinations relative both to operation-specific untied modules and to trained domain-generalist procedural modules?

---

## 1. Frozen design summary

Upon approval, OPM v1 uses:

- three surface domains: `SET`, `SCENE`, and `PROGRAM`;
- four query relations: `WITHIN`, `BEFORE`, `SAME`, and `LINK`, with auxiliary fact relations `DIRECT_IN` and `NESTED_IN`;
- four primitive operation tokens: `LOOKUP`, `REVERSE`, `CHAIN`, and `LIFT`;
- binary true/false queries;
- one-step and two-step procedures with oracle argument bindings and oracle execution order;
- eight unlabeled primitive modules, of which four are selected through a fixed per-seed permutation and four are unused sentinels in Stage A;
- separate two-layer fact and query encoders rather than an unrestricted full-world transformer;
- a primitive that may read exactly one oracle-selected fact per call;
- a fixed two-step executor, with a `STOP` mask after step one for one-step tasks;
- a shared OPM condition, a domain-generalist procedural control, and a domain-untied operation-specific control;
- deterministic synthetic datasets with sealed primary split seeds;
- five training seeds per primary condition;
- and paired bootstrap inference over the same held-out latent worlds.

No semantic identity is assigned to a primitive module index across seeds. The fixed operation-to-module permutation is stored in every run manifest.

---

## 2. Decision log

| ID | Status | Decision | Rationale | Requires owner approval |
|---|---|---|---|---:|
| DEC-001 | Proposed | Use query relations `WITHIN`, `BEFORE`, `SAME`, `LINK`; use `DIRECT_IN` and `NESTED_IN` only as facts | Separates direct membership from derived containment closure | Yes |
| DEC-002 | Proposed | Use `SET`, `SCENE`, `PROGRAM` as surface domains | Provides deliberately distinct renderers while retaining exact latent semantics | Yes |
| DEC-003 | Proposed | Primitive sees one selected fact per call | Prevents one module from solving an entire two-fact composition | Yes |
| DEC-004 | Proposed | Fixed per-seed one-to-one map for four operation tokens; four sentinel modules unused | Preserves oracle routing and tests unused-module specificity | Yes |
| DEC-005 | Proposed | OPM v1 excludes learned routing and learned binding | Isolates procedural sharing before decomposition learning | Yes |
| DEC-006 | Proposed | Primary endpoint requires OPM to beat a trained domain-generalist; untied operation modules remain a transfer diagnostic | Avoids comparing only a trained shared module with an untrained held-out copy | Yes |
| DEC-007 | Proposed | Five model seeds and paired world-level bootstrap | Practical initial power with paired evaluation | Yes |
| DEC-008 | Proposed | No unrestricted residual | Removes the primary bypass channel in exact-isomorphism testing | Yes |
| DEC-009 | Proposed | Remove H4 and qualifier-conditioned behavior from v1 | The original exception test provided no training basis for qualifier semantics | Yes |
| DEC-010 | Proposed | Attach typed canonical endpoint embeddings to every selected fact | Makes cross-domain state/evidence interchange structurally possible under oracle decomposition | Yes |
| DEC-011 | Proposed | Generate one size-three `SAME` class and use full algebraic closure for labels | Enables meaningful composition and prevents false negative labels | Yes |

Approval of this specification approves these decisions only for OPM v1.

---

## 3. Artifact A — typed relational algebra

### ALG-001 Entity types

The latent world contains two disjoint entity types:

```text
OBJECT     atomic entity
CONTAINER  entity capable of containing OBJECT or CONTAINER
```

Each entity has a unique integer ID within a world. Surface names are renderer-specific and randomized.

### ALG-002 Relations

| Relation | Signature | Symmetric | Reflexive | Transitive | Meaning |
|---|---|---:|---:|---:|---|
| `WITHIN` | `(OBJECT, CONTAINER)` | No | No | No | Query relation: object is directly or derivationally within container |
| `BEFORE` | `(OBJECT, OBJECT)` | No | No | Yes | Strict partial order |
| `SAME` | `(OBJECT, OBJECT)` | Yes | Yes | Yes | Equivalence relation |
| `LINK` | `(OBJECT, OBJECT)` | Yes | No | No | Undirected direct link |

`WITHIN` is a query relation and is never serialized as a world fact. It is derived from the auxiliary fact relations below. This prevents one predicate from ambiguously meaning both direct and inherited membership.

### ALG-003 Auxiliary fact relation

The world may contain:

```text
DIRECT_IN(OBJECT x, CONTAINER c)
NESTED_IN(CONTAINER child, CONTAINER parent)
```

`DIRECT_IN` is asymmetric and irreflexive. `NESTED_IN` is asymmetric, irreflexive, and acyclic. Neither is a query target in v1.

The only v1 derivation rules for `WITHIN` are:

\[
DIRECT\_IN(x,c) \Rightarrow WITHIN(x,c)
\]

\[
DIRECT\_IN(x,c) \land NESTED\_IN(c,p) \Rightarrow WITHIN(x,p)
\]

No closure beyond one `NESTED_IN` edge is queried in v1.

### ALG-004 Query targets

Every target is binary:

```text
QUERY(relation, role_1_entity, role_2_entity) -> {FALSE, TRUE}
```

The query signature must match the target relation. Ill-typed queries are excluded rather than labeled false.

### ALG-005 Primitive operation semantics

The dataset exposes four abstract operation tokens. They describe computation, not answer labels.

#### `LOOKUP`

Input: one fact `R_f(a,b)` and query `R_q(x,y)`. Usually `R_f=R_q`. For membership lookup, `R_f=DIRECT_IN` and `R_q=WITHIN`.  
Output evidence: true iff the relation pair is allowed by this rule and `(a,b)=(x,y)`.

#### `REVERSE`

Input: one fact `R(a,b)` where `R` is declared symmetric, and query `R(b,a)`.  
Output evidence: true. Negative examples use a fact whose unordered entity pair differs from the query pair.

#### `CHAIN`

Step 1 input: `R(a,b)` for `R in {BEFORE,SAME}`. State becomes `(R,a,b,valid=1)`.  
Step 2 input: `R(b,c)`. Output evidence supports query `R(a,c)`.  
The middle entity must match exactly. A mismatched middle entity produces false evidence.

#### `LIFT`

Step 1 input: `DIRECT_IN(x,c)`. State becomes `(x,c,valid=1)`.  
Step 2 input: `NESTED_IN(c,p)`. Output evidence supports query `WITHIN(x,p)`.

### ALG-006 Procedure table

| Query family | Steps | Oracle tokens | Positive condition |
|---|---:|---|---|
| Direct | 1 | `LOOKUP, STOP` | selected fact exactly matches ordered query |
| Symmetry | 1 | `REVERSE, STOP` | selected `SAME` or `LINK` fact matches reversed query |
| Transitive | 2 | `CHAIN, CHAIN` | two `BEFORE` or `SAME` facts share the required middle entity |
| Membership lift | 2 | `LIFT, LIFT` | `DIRECT_IN(x,c)` followed by `NESTED_IN(c,p)` entails `WITHIN(x,p)` |

`STOP` is an execution mask, not a selectable primitive.

### ALG-007 Negative construction

Every positive template has a matched negative formed by exactly one structural corruption:

- replace one queried endpoint with a type-compatible entity;
- reverse an asymmetric ordered pair;
- break the chain middle entity;
- replace the second relation with a signature-compatible distractor;
- or select a nonmatching direct fact.

The corruption type is balanced within each procedure family. No negative is accepted if the full latent world entails the query under REN-001C, regardless of the model's two-step execution limit.

### ALG-008 World validity

A world is valid only if:

- IDs are unique;
- entity types satisfy all relation signatures;
- `BEFORE` is acyclic;
- `NESTED_IN` is acyclic and has no queried path longer than one edge;
- `SAME` components form disjoint equivalence classes;
- `LINK` contains no self-edge or duplicate undirected edge;
- no generated negative query is entailed under the complete relation-specific semantics in REN-001C;
- and every query has exactly one declared oracle procedure.

The generator must run a symbolic verifier before serialization.

---

## 4. Artifact B — generators and renderers

### REN-001 Latent-world generator

The canonical RNG is NumPy `Generator(PCG64DXSM(seed_uint64))`. `derive_uint64(parts...)` joins UTF-8 string representations with `/`, computes SHA-256, takes digest bytes 0–7, and interprets them as an unsigned big-endian integer. Every sampling operation below consumes its declared stream in written order. Collection iteration is sorted before any random choice. Rejection restarts from a derived attempt seed `derive_uint64(world_seed,"attempt",n)`; the maximum is 100 attempts, after which generation fails loudly.

Executable pseudocode for a seed `s`, object count `n_o`, and container count `n_c`:

```text
for attempt in 0..99:
  rng = Generator(PCG64DXSM(derive_uint64(s,"attempt",attempt)))
  objects = [0..n_o-1]
  containers = [n_o..n_o+n_c-1]

  # BEFORE: exact edge count min(n_o, 8)
  topo = rng.permutation(objects)
  candidates = sorted((topo[i],topo[j]) for i<j)
  before = first min(n_o,8) pairs from rng.permutation(candidates)

  # SAME: one size-three class; serialize a directed two-edge chain only
  same_members = rng.choice(objects,size=3,replace=false)
  same_classes = [{same_members[0],same_members[1],same_members[2]}]
  ordered_same = sorted(same_classes[0])
  same_facts = []
  same_facts.extend([(ordered_same[0],ordered_same[1]),
                     (ordered_same[1],ordered_same[2])])

  # LINK: exact edge count min(n_o,8)
  link_candidates = sorted((a,b) for a<b in objects)
  link = first min(n_o,8) pairs from rng.permutation(link_candidates)

  # DIRECT_IN: exactly one per object
  direct_in = [(x, rng.choice(containers)) for x in sorted(objects)]

  # NESTED_IN: forest of depth one for queried edges
  root = rng.choice(containers)
  nested_candidates = [c for c in containers if c != root]
  nested_in = [(c,root) for c in sorted(nested_candidates)]

  facts = canonical_sort(DIRECT_IN,direct_in;
                         NESTED_IN,nested_in;
                         BEFORE,before;
                         SAME,same_facts;
                         LINK,link)
  world = World(objects,containers,facts,same_classes)
  positives = enumerate_templates_exactly(world)
  negatives = []
  for p in canonical_sort(positives):
    candidates = enumerate_single_corruptions(world,p)
    candidates = filter(not_entailed_under_full_semantics, candidates)
    if candidates empty: reject attempt
    negatives.extend(candidates)
  if world passes ALG-008: return world, positives, negatives
raise GenerationError(seed=s, attempts=100)
```

Training structural sizes vary by independently sampling 6–10 objects and 3–5 containers. Validation and ordinary test sizes use the same range. The structural-generalization test uses 12–16 objects and 6–8 containers.

For each world, `n_o` and `n_c` are sampled uniformly over their inclusive ranges before attempt zero using a separate size stream derived from `"<world-seed>/size"`.

### REN-001A Example allocation

For each split, enumerate examples in canonical order `(world_id, procedure_family, relation, label, canonical_query)`. Allocate exact equal quotas to each permitted `(domain, operation, label)` cell. Within a cell, select examples by sorting on `SHA256("<split>/<example-canonical-json>")` and taking the first required count. If a cell lacks its quota, deterministically generate additional worlds using the next world index; if the declared maximum world count is exceeded, generation fails rather than resampling another cell.

Renderer variants 0 and 1 alternate by the low bit of `SHA256("renderer/<split>/<example-id>")`, yielding an exact difference of at most one example per cell. Variant 2 occurs only in `Test-renderer`.

Duplicate latent derivations producing the same `(facts, query, label, procedure)` are canonicalized to one example before allocation. For `SAME`, only the directed chain edges explicitly listed above are serialized; closure membership is computed symbolically and is not serialized as all pairs.

### REN-001B Canonical ordering

`canonical_sort` compares tuples lexicographically after mapping relation IDs in this order:

```text
DIRECT_IN=0, NESTED_IN=1, BEFORE=2, SAME=3, LINK=4
```

Entity IDs are compared numerically. Queries sort by `(query_relation_id, arg1, arg2)`, with query relation IDs `WITHIN=0, BEFORE=1, SAME=2, LINK=3`. Procedures sort `LOOKUP=0, REVERSE=1, CHAIN=2, LIFT=3`. Canonical JSON uses sorted keys, no insignificant whitespace, and array order preserved.

### REN-001C Full semantic entailment

`entailed(world, query)` is defined exactly:

```text
if query is WITHIN(x,c):
  return DIRECT_IN(x,c) exists
         or exists m: DIRECT_IN(x,m) and NESTED_IN(m,c)

if query is BEFORE(a,b):
  return b is reachable from a by one or more BEFORE edges
         using deterministic breadth-first search over ascending neighbor IDs

if query is SAME(a,b):
  return a and b are in the same connected component of the undirected graph
         formed from serialized SAME edges; a==b is true

if query is LINK(a,b):
  return unordered_pair(a,b) is a serialized LINK edge
```

`not_entailed_under_full_semantics(candidate)` returns `not entailed(candidate.world,candidate.query)`. The model execution limit of two steps never limits symbolic target semantics.

### REN-001D Positive-template enumeration

`enumerate_templates_exactly(world)` returns this union, with duplicates removed by canonical example key:

```text
LOOKUP positives:
  for each BEFORE(a,b): query BEFORE(a,b), evidence [that fact]
  for each SAME(a,b): query SAME(a,b), evidence [that fact]
  for each LINK(a,b): query LINK(a,b), evidence [that fact]
  for each DIRECT_IN(x,c): query WITHIN(x,c), evidence [that fact]

REVERSE positives:
  for each SAME(a,b) with a!=b: query SAME(b,a), evidence [SAME(a,b)]
  for each LINK(a,b): query LINK(b,a), evidence [LINK(a,b)]

CHAIN positives:
  for each ordered distinct triple (a,b,c) where serialized facts
      BEFORE(a,b) and BEFORE(b,c) exist:
    query BEFORE(a,c), evidence [BEFORE(a,b), BEFORE(b,c)]
  for each ordered distinct triple (a,b,c) where serialized facts
      SAME(a,b) and SAME(b,c) exist:
    query SAME(a,c), evidence [SAME(a,b), SAME(b,c)]

LIFT positives:
  for each DIRECT_IN(x,c) and NESTED_IN(c,p):
    query WITHIN(x,p), evidence [DIRECT_IN(x,c), NESTED_IN(c,p)]
```

Facts referenced by evidence are located by their indices after the final per-example fact shuffle; the latent template stores fact identities until rendering.

### REN-001E Single-corruption enumeration

`enumerate_single_corruptions(world,p)` creates candidates in the following ordered groups. Each candidate retains the same operation tokens and step count.

```text
endpoint corruption:
  replace query arg2 with every other entity of the required type, ascending ID

order corruption:
  if query relation is asymmetric, swap query arguments when the signature remains valid

middle corruption for CHAIN or LIFT:
  replace the first argument of evidence step 2 with every type-compatible entity
  except the required middle entity, ascending ID; use an existing fact when available,
  otherwise the candidate is not emitted

relation corruption:
  replace one evidence fact with each existing fact having the same argument-type
  signature but a different relation, ordered by fact index

direct mismatch for LOOKUP or REVERSE:
  replace evidence step 1 with every existing same-relation fact whose endpoints
  do not satisfy the query under that procedure, ordered by fact index
```

After enumeration:

1. remove candidates whose query is entailed under REN-001C;
2. remove candidates identical to the positive canonical key;
3. deduplicate by `(facts,query,operation_tokens,evidence_fact_identities)`;
4. canonical-sort by `(corruption_type,query,evidence_fact_identities)`;
5. retain all remaining candidates for split-level allocation.

If a corruption group is inapplicable, it contributes no candidate. No per-world random negative is selected.

For each negative `(split,domain,operation,label=0)` allocation cell, collect candidates from successive worlds until every corruption type that is valid for that operation can meet its quota. Let `types` be the lexicographically sorted valid corruption types and `N` the cell size. Assign quota `floor(N/len(types))` to each type, then assign the first `N mod len(types)` types one additional slot. Within each type, rank candidates by `SHA256("negative-rank/<split>/<domain>/<operation>/<canonical-candidate-json>")`, deduplicate, and take the first quota items. If any quota remains unmet at the split's maximum world index, generation fails. Positive cells are allocated independently under REN-001A.

Thus the allocator generates additional worlds rather than silently substituting a different corruption type or searching farther within a positive example's candidates.

### REN-002 Canonical sample record

Before surface rendering, every example is stored as:

```yaml
world_id: uint64
example_id: 32-char lowercase hex
domain: SET | SCENE | PROGRAM
renderer_variant: uint8
facts:
  - [relation_id, role_1_entity_id, role_2_entity_id]
query: [relation_id, role_1_entity_id, role_2_entity_id]
label: 0 | 1
oracle:
  argument_slots: [role_1_entity_id, role_2_entity_id]
  evidence_indices: [fact_index_1, fact_index_2_or_PAD]
  operation_tokens: [op_1, op_2_or_STOP]
  step_count: 1 | 2
metadata:
  procedure_family: direct | symmetry | transitive | lift
  corruption_type: none | endpoint | order | middle | relation | direct_mismatch
```

`example_id` is the first 128 bits of `SHA256(canonical_json({world_id,domain,renderer_variant,query,operation_tokens,evidence_fact_identities,corruption_type}))`, encoded as 32 lowercase hexadecimal characters. The label is intentionally excluded because it is already determined by the structural fields.

Evidence indices are provided to both the shared and untied procedural models. This makes v1 an execution-and-sharing test rather than an evidence-retrieval test.

### REN-003 `SET` renderer

For every domain, surface renaming uses `Generator(PCG64DXSM(derive_uint64(world_id,"rename",domain)))`. Object and container token inventories are independently permuted with NumPy `permutation`; latent entities in ascending ID order are assigned tokens in permuted order. This mapping is stored in audit metadata and supplies the canonical fact endpoint IDs in MOD-004, but the mapping itself is not an unrestricted model input.

Surface vocabulary:

```text
objects: e00 ... e31, randomly permuted per world
containers: s00 ... s15, randomly permuted per world
relations:
  DIRECT_IN   -> member(e,s)
  NESTED_IN   -> subset(s_child,s_parent)
  BEFORE      -> ordered_before(e1,e2)
  SAME        -> equivalent(e1,e2)
  LINK        -> paired(e1,e2)
  query WITHIN  -> ask:within(e,s)
  other queries -> ask:<relation>(arg1,arg2)
```

Facts are separated by `;`. Variant 0 uses relation-first notation. Variant 1 uses tuple-first notation such as `(e1,e2):ordered_before`. Fact order is uniformly shuffled.

### REN-004 `SCENE` renderer

Surface vocabulary uses object tokens `o00...o31` and region tokens `r00...r15`.

```text
  DIRECT_IN   -> o @ r
NESTED_IN   -> r_child << r_parent
BEFORE      -> o1 <t o2
SAME        -> o1 == o2
LINK        -> o1 -- o2
query WITHIN -> ? WITHIN object region
other query  -> ? relation arg1 arg2
```

Variant 0 is infix. Variant 1 serializes each fact as `[arg1, relation_symbol, arg2]`. This is a symbolic scene graph, not a pixel geometry task.

### REN-005 `PROGRAM` renderer

Surface vocabulary uses variables `v00...v31` and collections `c00...c15`.

```text
  DIRECT_IN   -> has(c, v)          # argument order differs on surface only
NESTED_IN   -> imports(c_parent, c_child)
BEFORE      -> precedes(v1, v2)
SAME        -> alias(v1, v2)
LINK        -> connected(v1, v2)
  query WITHIN-> assert? within(v,c)
  other query -> assert? function(arguments)
```

The canonical role order is restored by the oracle binding record. Variant 1 uses keyword arguments in randomized order. This deliberate surface-role change tests adapter compatibility without changing latent semantics.

### REN-006 Distractors

Canonical example fact lists contain exactly 8 facts for train, validation, interpolation, recombination, and renderer splits, and exactly 12 for structural testing. Required evidence occupies one or two latent slots before shuffling; all remaining slots are distractors selected from serialized world facts not already used as evidence.

Distractor selection is deterministic:

```text
candidates = canonical_sort(world_facts - evidence_facts)
shared = [f for f in candidates if either endpoint occurs in the query]
other  = [f for f in candidates if f not in shared]
rank each list by SHA256("distractor/<example-id>/<canonical-fact>")
take the first min(2,required_distractors) from shared
then repeatedly cycle relation order
  [DIRECT_IN,NESTED_IN,BEFORE,SAME,LINK]
and take the highest-ranked remaining fact of that relation
until the required count is reached
if a full cycle adds nothing, take the highest-ranked remaining fact regardless of relation
if insufficient candidates remain, reject the world/example template
```

After selection, evidence and distractors are shuffled together by sorting on `SHA256("fact-order/<example-id>/<canonical-fact>/<occurrence-index>")`. Oracle evidence indices are computed after this shuffle. The query and label do not enter distractor ranking except through the opaque example ID. No surface feature marks evidence status.

### REN-007 Held-out renderer

Renderer variant 2 is test-only and fully deterministic. It uses prefix records:

```text
fact  := [ alias , surface_arg1 , surface_arg2 ]
query := [ ? , alias , surface_arg1 , surface_arg2 ]
```

Commas and brackets are literal lexer tokens. Facts appear in the reverse of the REN-006 hash-sorted fact order. Entity renaming uses the ordinary domain mapping rotated left by:

```text
rotation = derive_uint64("opm-v1","renderer-v2",world_id,domain) mod inventory_size
```

separately for object and container inventories.

The fixed test-only semantic-to-alias maps are:

| Latent relation/query | SET alias | SCENE alias | PROGRAM alias |
|---|---|---|---|
| `DIRECT_IN` | `paired` | `--` | `connected` |
| `NESTED_IN` | `member` | `@` | `has` |
| `BEFORE` | `subset` | `<<` | `imports` |
| `SAME` | `ordered_before` | `<t` | `precedes` |
| `LINK` | `equivalent` | `==` | `alias` |
| `WITHIN` query | `within` | `WITHIN` | `within` |

This is a fixed alias permutation, not a random per-example substitution. It deliberately reverses familiar surface associations while preserving argument arity, canonical endpoint bindings, and oracle procedure information. All aliases and delimiters already occur in the frozen vocabulary; variant 2 introduces no new token IDs.

---

## 5. Artifact C — dataset manifest

### DAT-001 Split sizes

Counts are examples after domain rendering.

| Split | Maximum world indices | Examples per domain | Domains | Purpose |
|---|---:|---:|---:|---|
| Train | 100,000 | 48,000 | 3 | optimization; 8,000 per permitted operation-label cell |
| Validation | 20,000 | 4,800 | 3 | model selection; 800 per permitted operation-label cell |
| Test-interpolation | 20,000 | 4,800 | 3 | observed cells only; 800 per permitted operation-label cell |
| Test-recombination | 20,000 | 6,000 | 3 | primary H1 endpoint; 3,000 per label in each domain's held-out operation |
| Test-renderer | 20,000 | 3,200 | 3 | 400 per operation-label cell |
| Test-structural | 20,000 | 3,200 | 3 | 400 per operation-label cell |

Each split is exactly balanced within the operation-label cells stated above. Relation composition within `CHAIN` is divided equally between `BEFORE` and `SAME`; any odd remainder is assigned to `BEFORE`. `LOOKUP` is divided as equally as possible across its four query relations, with remainder assigned in lexicographic relation order. Balance is achieved through deterministic allocation, not stochastic subsampling.

`Test-interpolation` excludes the three held-out domain-operation cells. `Test-recombination` contains only those cells. `Test-renderer` and `Test-structural` contain all cells but must report observed and held-out strata separately; they are secondary robustness tests and do not enter the primary H1 estimate.

### DAT-002 Recombination holdout

Training withholds these domain-operation pairs:

```text
SET     × REVERSE
SCENE   × LIFT
PROGRAM × CHAIN
```

All four operation tokens occur in training, and every domain occurs with at least three operation families. `LOOKUP` is trained in all domains to ensure every adapter receives direct supervision.

The primary recombination test contains only the three withheld pairs, equally weighted.

### DAT-003 Seed namespaces

Seeds use `derive_uint64` from REN-001:

```text
opm-v1/train/world/<index>
opm-v1/validation/world/<index>
opm-v1/test-interpolation/world/<index>
opm-v1/test-recombination/world/<index>
opm-v1/test-renderer/world/<index>
opm-v1/test-structural/world/<index>
```

Model run seeds are `1101, 2202, 3303, 4404, 5505`.

### DAT-004 Fingerprints

Each JSONL split is canonicalized with sorted keys and UTF-8 LF endings. Its manifest records row count, SHA-256, generator revision, renderer revision, seed namespace, class counts, and holdout matrix.

Primary test labels must be stored separately from model-readable inputs and accessed only by the locked evaluator after protocol freeze.

### DAT-005 Contamination checks

The generator must assert:

- no repeated `world_id` across splits;
- no identical canonical fact/query tuple across splits;
- no renderer-variant-2 sample in training;
- no recombination holdout pair in training;
- and no undeclared relation or renderer variant in canonical training.

---

## 6. Artifact D/E — tensor contracts and primitive boundary

### MOD-001 Dimensions

```yaml
dimensions:
  d_model: 192
  d_entity: 64
  d_relation: 32
  d_domain: 16
  d_state: 192
  d_hidden_primitive: 384
  max_facts_train: 8
  max_facts_test: 12
  primitive_count: 8
  selected_primitive_count_per_step: 1
  max_steps: 2
```

All learned tensors use `float32` in canonical v1 to prioritize determinism. Masks are Boolean. IDs are `int64`.

### MOD-002 Fact encoding

Each surface fact is parsed by its renderer-specific deterministic tokenizer into surface tokens. A shared token embedding and two-layer, four-head transformer fact encoder produce one vector per fact:

```text
fact_tokens:       [B,F,T] int64
fact_token_mask:   [B,F,T] bool
fact_vectors:      [B,F,192] float32
```

Facts are encoded independently; self-attention does not cross fact boundaries. This prevents the encoder from performing the entire two-fact inference before primitive execution.

The fact vector is the final-layer hidden state at the leading `[FACT]` token. Attention padding positions are masked to negative infinity before softmax.

### MOD-002A Encoder block

Fact and query encoders use the same architecture with separate parameters:

```text
input = token_embedding[token_id] + learned_position_embedding[position]
for layer in 1..2:
  x = x + Dropout(MHA(LayerNorm(x), heads=4, head_dim=48))
  x = x + Dropout(Linear(768,192)(GELU(Linear(192,768)(LayerNorm(x)))))
output = LayerNorm(x)
```

Attention uses scaled dot-product softmax, four independent query/key/value projections with combined implementation permitted, output projection 192→192, and attention-probability dropout equal to the global dropout. Learned position embeddings have shape `[12,192]`. There is no cross-fact attention, segment embedding, relative position term, rotary embedding, or causal mask.

### MOD-003 Query and context

```text
query_tokens:      [B,Tq] int64
query_vector:      [B,192] float32
domain_id:         [B] int64
domain_vector:     [B,16] float32
argument_entity_ids:[B,2] int64
```

The query encoder is a two-layer, four-head transformer. It shares token embeddings but not transformer weights with the fact encoder. The query vector is the final-layer hidden state at the leading `[QUERY]` token.

`domain_vector` is produced by `Embedding(3,16)` with IDs `SET=0`, `SCENE=1`, and `PROGRAM=2`.

### MOD-003A Vocabulary and sequence limits

The lexer emits whole reserved symbols and ASCII identifiers. Token IDs are assigned in this exact order:

1. `[PAD]=0`, `[FACT]=1`, `[QUERY]=2`, `[UNK]=3`;
2. punctuation and operator tokens in ascending Unicode code-point order;
3. reserved relation/function words in ASCII lexicographic order;
4. object identifiers `e00..e31`, `o00..o31`, `v00..v31` in the written order;
5. container identifiers `s00..s15`, `r00..r15`, `c00..c15` in the written order.

The complete generated vocabulary must be frozen as `vocabulary.json`. Maximum fact length is 12 tokens including `[FACT]`; maximum query length is 12 including `[QUERY]`. Short sequences right-pad with ID 0. Longer sequences are a generator error and must never be truncated.

The lexer grammar is applied left-to-right with longest-token priority:

```ebnf
input       = { whitespace | token } ;
whitespace  = " " | "\t" | "\r" | "\n" ;
token       = multi_symbol | punctuation | identifier ;
multi_symbol= "<<" | "<t" | "==" | "--" ;
punctuation = "(" | ")" | "[" | "]" | "," | ";" | ":" | "?" | "@" ;
identifier  = letter , { letter | digit | "_" } ;
letter      = "A".."Z" | "a".."z" ;
digit       = "0".."9" ;
```

Whitespace is discarded. Identifiers are case-sensitive and must occur in the frozen vocabulary. Any unmatched character or unknown identifier is a serialization error; `[UNK]` exists for diagnostic robustness tests but is never emitted by canonical data. Variant 2 uses the fixed alias table in REN-007; every alias must already occur in the frozen vocabulary.

### MOD-003B Entity embeddings

Oracle entity IDs are converted to within-world typed ordinals. Objects use indices `0..31`; containers use `32..47`; index `48` is PAD. A learned table `EntityEmbedding(49,64)` is shared across domains and worlds. Because renderers independently rename entities, the renderer record includes the bijection from surface token to typed ordinal. This bijection is available only through the oracle argument slots and deterministic rendering audit, not as an additional model input.

### MOD-004 Oracle evidence read

At step `t`, the executor gathers exactly one surface vector and the selected fact's two canonical typed endpoint IDs:

```text
evidence_index: [B] int64
surface_e_t:    [B,192] float32 = gather(fact_vectors, evidence_index)
fact_endpoints: [B,2] int64 = gather(canonical_fact_endpoint_ids, evidence_index)
endpoint_pair:  [B,128] float32 = concat(EntityEmbedding(endpoint_1),
                                         EntityEmbedding(endpoint_2))
e_t:            [B,192] float32 = LayerNorm(surface_e_t + Wef endpoint_pair)
```

`Wef` is a learned `Linear(128,192)` with initialization from MOD-005A. Endpoint order is the canonical latent relation-role order, even when a renderer reverses surface argument order. This is oracle decomposition information and is supplied identically to every neural control.

The primitive cannot access unselected facts, the full fact tensor, raw labels, future evidence indices, or the renderer's mapping for any unselected fact. Canonical selected-fact bindings make cross-domain interchange possible without exposing the answer.

### MOD-005 State initialization

```text
s_0 = LayerNorm(Wq query_vector
              + Wd domain_vector
              + Wa concat(entity_embedding[arg1], entity_embedding[arg2])
              )
```

All projections output 192 dimensions. `Wq` and `Wd` include biases; `Wa` maps 128 to 192 with bias. Global entity identity is not shared across worlds.

### MOD-005A Initialization, normalization, and dropout

- Embedding and linear weights use PyTorch `trunc_normal_(mean=0,std=0.02,a=-0.04,b=0.04)`.
- Linear biases are zero.
- LayerNorm scale is one and bias is zero with epsilon `1e-5`.
- Transformer attention and feed-forward residual outputs use dropout `p` from TRN-004.
- Primitive `h_t` receives dropout after GELU and before `W2_k`.
- State residual, gates, logits, embeddings, and final LayerNorm receive no additional dropout.
- The same model seed initializes corresponding non-primitive components identically across primary conditions.

### MOD-006 Primitive equation

Each primitive `C_k` is a two-layer gated MLP transition:

```text
u_t = LayerNorm(concat(s_t, e_t))                 # [B,384]
h_t = GELU(W1_k u_t + b1_k)                      # [B,384]
g_t = sigmoid(Wg_k u_t + bg_k)                   # [B,192]
delta_t = W2_k h_t + b2_k                        # [B,192]
s_(t+1) = LayerNorm(s_t + g_t * delta_t)         # [B,192]
```

Primitive parameters are the only parameters tied or untied in the shared-versus-untied diagnostic. Encoders, initialization projections, entity embeddings, and decoder are shared across domains in all neural conditions. `DOMAIN_GENERALIST` additionally contains the shared operation embedding and the wider input projection defined in MOD-011.

### MOD-007 Execution

For each step:

1. read the oracle operation token;
2. map it through the fixed per-seed permutation to one primitive index;
3. gather the oracle-selected evidence vector;
4. execute exactly that primitive;
5. apply `STOP` after step one when `step_count=1`.

No learned route revision, parallel mixture, recurrence beyond two steps, or access to all modules is allowed in canonical sparse Stage A.

For one-step tasks, the second evidence index is integer `-1` in serialized data. The loader converts it to gather index zero only after setting `step_mask[:,1]=false`; the gathered value is overwritten with an all-zero vector and no second primitive executes. `[PAD]` entity index 48 is used only for absent roles in diagnostics, not canonical binary queries.

### MOD-008 Decoder

```text
answer_logits = Linear(192,2)(LayerNorm(s_final))
```

The decoder sees only final state. It does not receive oracle tokens, evidence indices, raw facts, or domain ID through a skip connection.

### MOD-009 Domain-untied condition

The untied executor contains `C_(d,k)` for each domain `d`. The fixed oracle permutation selects operation `o -> k`, then the domain selects the corresponding untied copy. Architecture and active primitive FLOPs are identical to the shared condition. Total primitive parameters are three times larger.

### MOD-010 One-step limitation

A primitive receives only one fact vector per call. Two-step positive and negative examples are constructed so neither selected fact alone predicts the label above chance after balancing. This must be verified by evidence-step-only probes before primary runs.

### MOD-011 Domain-generalist control

`DOMAIN_GENERALIST` contains one transition module `G_d` per domain rather than one module per operation. Every observed operation in domain `d` updates the same `G_d`. A globally shared learned operation embedding `r_o in R^32` is supplied at each step:

```text
u_t = LayerNorm(concat(s_t, e_t, r_o))            # [B,416]
h_t = GELU(W1_d u_t + b1_d)                       # [B,384]
g_t = sigmoid(Wg_d u_t + bg_d)                    # [B,192]
delta_t = W2_d Dropout(h_t) + b2_d                # [B,192]
s_(t+1) = LayerNorm(s_t + g_t * delta_t)          # [B,192]
```

`r_o` is shared across domains and receives gradients wherever operation `o` is observed. Thus, on a withheld domain-operation cell, `G_d` is trained on the other operations in that domain and `r_o` is trained in other domains. This deliberately strong control can generalize without an operation-specific shared module.

Because its input is 32 dimensions wider, `DOMAIN_GENERALIST` is capacity-advantaged. Its actual parameter and FLOP ratios must be reported; it is not described as exactly matched.

---

## 7. Artifact F — oracle contract

### ORC-001 Vocabulary

```text
operation tokens: LOOKUP, REVERSE, CHAIN, LIFT
execution mask:   STOP, CONTINUE
```

Operation tokens do not encode relation, domain, entity, corruption, or label.

### ORC-002 Fixed routing

For model seed `s`:

```text
rng = Generator(PCG64DXSM(derive_uint64("opm-v1","module-permutation",s)))
active_indices = FisherYatesShuffle([0,1,2,3,4,5,6,7], rng)[0:4]
map sorted operation tokens [CHAIN,LIFT,LOOKUP,REVERSE]
to active_indices in order
```

The remaining four modules are sentinel modules and receive no canonical training gradient. The map is saved in the run manifest.

### ORC-003 Bindings

Oracle argument slots contain only the two entity IDs already present in the query, in canonical semantic role order. They do not identify the answer or the relation beyond what the query surface already contains.

### ORC-004 Evidence

Evidence indices identify one fact per execution step. Both positive and negative examples receive structurally analogous evidence indices. The oracle selects the facts to inspect but does not state whether they compose successfully.

### ORC-005 Raw oracle-channel leakage gate

Train three probes using training data and evaluate on validation:

1. operation tokens plus step count only;
2. evidence indices plus fact positions only;
3. oracle argument IDs after within-world random renaming, without facts.

Each logistic probe uses scikit-learn `LogisticRegression` with:

```yaml
solver: lbfgs
penalty: l2
C: 1.0
fit_intercept: true
max_iter: 2000
tol: 1.0e-8
class_weight: null
random_state: 91001
feature_scaling: StandardScaler fit on probe training only
multiclass: auto
```

Categorical tokens and indices are one-hot encoded with categories fitted on training and unknown validation categories ignored. Probe training uses the canonical model-training split; probe selection uses no hyperparameter search. Validation accuracy must be `<= 0.525` with a two-sided 95% Wilson interval containing `0.50`. Holm correction at familywise `alpha=0.05` is applied to the three oracle-channel probes. These raw probes run after dataset generation and before `PROTOCOL_FROZEN`; any failure blocks protocol freeze.

For each probe, the hypothesis test is an exact one-sided binomial test over validation predictions:

```text
H0: probability(correct) <= 0.50
H1: probability(correct) > 0.50
```

Compute the unadjusted exact-binomial p-value with `scipy.stats.binomtest(k,n,p=0.5,alternative="greater")`. Sort the three p-values ascending and apply Holm step-down thresholds `0.05/(3-i)` for zero-based rank `i`, stopping after the first non-rejection. A channel passes only if its accuracy threshold and Wilson-interval conditions hold and the Holm procedure **fails to reject** its null hypothesis. Rejection means evidence of above-chance leakage and is a failure, not a pass.

### ORC-006 Neural evidence-vector mechanism probes

Each individual evidence vector with query removed must yield accuracy `<= 0.55` on two-step tasks using a two-layer MLP probe with widths `192→128→2`, GELU, AdamW at `1e-3`, batch size 256, 100 epochs, no early stopping, and standardized input features.

These probes require a trained fact encoder and therefore do not gate protocol freeze. They run separately for:

```text
each neural model condition
× each declared model seed
× evidence step {1,2}
```

The probe seed is `derive_uint64("opm-v1","neural-probe",model_condition,model_seed,step)`. Probe training uses frozen training representations and evaluates frozen validation representations from the same trained run. The two step probes form one Holm family within each `(model_condition,model_seed)` run; no p-values are pooled across runs.

The two MLP probes use the same exact one-sided binomial test with a two-test Holm family and must also have two-sided 95% Wilson intervals containing `0.50`. A run-level neural probe passes only when both null hypotheses fail to reject and both descriptive thresholds pass. Failure invalidates the affected run's mechanism claim; it does not alter the frozen protocol, exclude the run from H1 accuracy analysis, or trigger retraining.

---

## 8. Artifact G — exact model configurations and baselines

### BAS-001 Primary models

| Model ID | Encoders | Primitives | Oracle | Execution | Primary role |
|---|---|---|---|---|---|
| `OPM_SHARED` | MOD-002/003 | 8, four active, continuously tied | Full Stage A | sparse sequential | proposed model |
| `PROC_UNTIED` | identical | 8 per domain, four active per domain | identical | sparse sequential | primary counterfactual |
| `PROC_CLONE` | identical | untied, identical initialization across domains | identical | sparse sequential | initialization control |
| `DOMAIN_GENERALIST` | identical | one trained transition per domain plus shared operation embeddings | identical | sparse sequential | primary generalization control |
| `ORACLE_SYMBOLIC` | none | programmed ALG rules | full | exact | data/upper-bound check |

`PROC_UNTIED` modules use independent initialization. `PROC_CLONE` copies the initial active-module weights from one seed-specific template and then trains independently.

`PROC_UNTIED` is a pure weight-transfer diagnostic because its held-out operation copy is untrained. H1 is not supported solely by defeating it. `DOMAIN_GENERALIST` is the principal scientific control because its domain transition is trained on every observed operation in that domain.

### BAS-002 Deferred secondary models

| Model ID | Purpose |
|---|---|
| `OPM_DENSE_EXEC` | Phase 2 sparsity control; deferred |
| `DENSE_TRANSFORMER` | later generic dense baseline; deferred |
| `VANILLA_MOE` | later learned-routing baseline; deferred |
| `ORACLE_MOE` | later oracle-informed routing baseline; deferred |

These models are explicitly out of scope for canonical v1 implementation and claims. They require a later specification amendment with exact inputs, equations, pooling, routing, output paths, parameter configurations, and FLOP matching. Artifact G's exact-configuration requirement applies in v1 only to `OPM_SHARED`, `PROC_UNTIED`, `PROC_CLONE`, `DOMAIN_GENERALIST`, and `ORACLE_SYMBOLIC`.

### BAS-003 Accounting

The implementation must generate, from traced forward graphs:

- exact trainable parameter count by component;
- active parameter count per one-step and two-step example;
- multiply-add count reported as two FLOPs;
- LayerNorm, activation, attention, gather, and routing estimates;
- and measured profiler time per component.

Matching tolerance for declared equal-active-FLOP comparisons is ±2%. If the tolerance is not achieved, results must be labeled unmatched and accompanied by the actual ratio.

---

## 9. Artifact H — losses

### LOS-001 Canonical training loss

Canonical v1 uses only answer cross-entropy:

\[
L_{task}=-\frac{1}{B}\sum_i \log p_\theta(y_i|x_i,o_i,b_i,e_i).
\]

This intentionally avoids adding interchange or counterfactual objectives before establishing whether architectural weight tying alone produces reuse.

### LOS-002 Optional auxiliary ablations

The following are exploratory and may not replace the canonical run:

#### Interchange loss

For paired examples derived from the same latent world and procedure in two domains, swap the state immediately before the final primitive call and apply answer cross-entropy to the swapped execution. The loss is the mean of both swap directions.

#### Counterfactual loss

For matched positive/negative pairs differing by one structural corruption, minimize ordinary cross-entropy on both and a margin loss:

\[
L_{cf}=\max(0,0.5-[\ell_{true}(x^+)-\ell_{true}(x^-)]).
\]

If enabled, use `L = L_task + 0.25 L_interchange + 0.10 L_cf` from step 5,000 onward. These coefficients are fixed for the exploratory ablation.

### LOS-003 No compute loss in Stage A

Stage A routes are fixed and always use one primitive per active step. A differentiable compute loss has no function and must not be added.

---

## 10. Artifact I — training protocol

### TRN-001 Optimization

```yaml
training:
  optimizer: AdamW
  beta1: 0.9
  beta2: 0.98
  epsilon: 1.0e-8
  weight_decay: 0.01
  base_learning_rate: 3.0e-4
  warmup_steps: 2000
  schedule: cosine_to_3.0e-5
  batch_size: 256
  gradient_accumulation: 1
  max_steps: 50000
  gradient_clip_norm: 1.0
  precision: float32
  label_smoothing: 0.0
  dropout: 0.1
  validation_every_steps: 500
  checkpoint_every_steps: 500
```

### TRN-002 Sampling

Training batches are balanced equally across domains, labels, and the four procedure families. Recombination holdout pairs are rejected before batch construction. One-step and two-step examples each occupy half the batch.

### TRN-003 Selection and stopping

The selected checkpoint is the one with highest macro validation accuracy across observed `(domain, operation)` cells. Ties within `0.001` choose the earlier checkpoint. Training stops at 50,000 steps; there is no adaptive early termination in canonical runs.

### TRN-004 Tuning budget

Before primary runs, one pilot seed may compare exactly:

```text
learning rate: {1e-4, 3e-4, 6e-4}
dropout:       {0.0, 0.1}
```

All model families receive the same six-combination budget. The selected global learning rate and dropout are then frozen for all primary models. Model-specific tuning is prohibited in v1.

---

## 11. Artifact J — metrics and interventions

### MET-001 Primary H1 metrics

For each model seed, compute macro accuracy across the three withheld `(domain, operation)` cells. The principal effect is:

\[
\Delta_{generalist}=Acc(OPM\_SHARED)-Acc(DOMAIN\_GENERALIST)
\]

The transfer-diagnostic effect is:

\[
\Delta_{untied}=Acc(OPM\_SHARED)-Acc(PROC\_UNTIED).
\]

Both use paired examples and matched model seeds. Only `Delta_generalist` may support H1; `Delta_untied` measures the expected benefit of training a shared operation rather than leaving a domain-operation copy untrained.

### MET-002 Secondary capability metrics

- interpolation macro accuracy;
- one-step and two-step accuracy;
- held-out renderer accuracy;
- structural-generalization accuracy;
- per-domain and per-operation accuracy;
- calibration error with 10 fixed-width bins.

### MET-003 Primitive ablation

For each active primitive, replace `delta_t` with zero at every selected occurrence. Report accuracy change by operation and domain. A sentinel ablation performs the same intervention on unused modules and should produce no numerical change.

### MET-004 Primitive replacement

Replace the selected active primitive with each other active primitive while keeping inputs fixed. Report the operation-by-replacement confusion matrix.

### MET-005 Cross-domain interchange

Pair examples from the same latent world, query, label, and procedure but different renderers. At the boundary before the last primitive call, swap normalized execution states while retaining the destination evidence vector. Report swapped accuracy and the drop from unswapped paired accuracy.

### MET-006 Adapter-only test

Zero all primitive deltas so final state contains only initialization and unmodified residual state. Report accuracy. Because there is no unrestricted residual, performance materially above chance indicates leakage through initialization or encoders.

### MET-007 Surface reversal

Use renderer variant 2 with relation symbols reassigned through a test-only permutation. Oracle bindings remain correct. Report the accuracy drop relative to the ordinary renderer test.

### MET-008 H4 exclusion

No exception or qualifier metric is defined in v1. H4 requires a later protocol in which qualifier meanings are learned in observed cells and only a target domain-qualifier combination is withheld.

---

## 12. Artifact K — preregistered statistics and thresholds

### STA-001 Analysis population

All five declared model seeds are included unless a run is corrupted by a predeclared infrastructure failure: nonfinite loss, unreadable checkpoint, dataset fingerprint mismatch, or evaluator failure. Low model quality is not an exclusion reason.

### STA-002 Primary test

Primitive parameter shapes do not enter pairing. Predictions are paired by the external key:

```text
(model_seed, world_id, domain, renderer_variant,
 operation, canonical_query, label, example_id)
```

Every primary model must emit exactly one prediction for every expected key. Missing or duplicate keys are evaluator failures, not silently dropped observations. Model seed `1101` is paired with seed `1101` across conditions, and likewise for all declared seeds; this pairs shared non-primitive initialization and data order even when primitive tensors differ.

Compute a two-level bootstrap:

1. resample model seeds with replacement;
2. within each selected seed, resample world IDs with replacement;
3. calculate `Delta_generalist` and `Delta_untied`;
4. repeat 10,000 times.

Use the percentile 95% confidence interval.

### STA-003 H1 decision

- `supported`: lower 95% bound of `Delta_generalist` is greater than `+0.02`, and OPM interpolation accuracy is no more than `0.01` below both `DOMAIN_GENERALIST` and `PROC_UNTIED`;
- `not_supported`: upper 95% bound of `Delta_generalist` is less than or equal to `+0.02`;
- `inconclusive`: lower bound is at most `+0.02` but upper bound is greater than `+0.02`, or either interpolation non-inferiority condition fails.

`Delta_untied` is reported with its own interval but has no independent support threshold.

### STA-004 Mechanism criteria

Evidence of reusable computation additionally requires all of:

- symbolic oracle accuracy equals `1.000`;
- shared-model recombination accuracy exceeds `0.80`;
- ablating the corresponding active primitive reduces its selected-operation accuracy by at least `0.20` in at least two trained domains;
- average unrelated-operation accuracy drop under that ablation is below `0.05`;
- sentinel ablation changes all logits by less than `1e-7`;
- adapter-only accuracy is at most `0.60`;
- raw oracle leakage probes pass ORC-005;
- and neural evidence-vector probes pass ORC-006 for every run used in the mechanism claim.

Failure of a mechanism criterion prevents the phrase “causally reusable primitive” even if H1 is supported.

### STA-005 H2 status

H2 remains exploratory. Report exact parameter and FLOP ratios; do not declare H2 supported unless a later protocol preregisters a quality-matched resource threshold.

### STA-006 H3 status

H3 is out of scope. Hardware timings are descriptive only.

### STA-007 H4 status

H4 is out of scope. No bounded-sharing conclusion may be drawn from v1.

---

## 13. Artifact L — environment and reproducibility

### SYS-001 Environment

```yaml
environment:
  language: Python 3.12
  framework: PyTorch 2.8
  device_priority: [CUDA, CPU]
  canonical_precision: float32
  deterministic_algorithms: true
  cudnn_benchmark: false
  compile_mode: disabled
  tokenizer: deterministic_whitespace_and_symbol_lexer
  data_format: canonical_jsonl
```

Exact patch versions must be captured from the installed lockfile or environment manifest at protocol freeze. If PyTorch 2.8 is unavailable, the specification must be amended before implementation; an agent must not silently substitute a version.

Workspace feasibility check on 2026-08-20: the existing `.venv` uses Python `3.12.10` and does not contain PyTorch. Approval therefore requires an explicit dependency and hardware decision; this draft does not install or alter the environment.

### SYS-002 Artifact structure

```text
opm/
  config/
  data/manifests/
  data/splits/
  models/
  training/
  interventions/
  evaluation/
  accounting/
  schemas/
artifacts/<run_id>/
  config.yaml
  environment.json
  dataset_manifest.json
  module_permutation.json
  checkpoints/
  metrics.json
  accounting.json
  events.jsonl
```

### SYS-003 Run identity

`run_id` is the SHA-256 prefix of canonical configuration, specification version, dataset fingerprints, model seed, and code revision. A run with any changed input receives a new ID.

### SYS-004 Required commands

The implementation must provide commands equivalent to:

```text
generate-data --spec OPM-V1-SPEC --verify
train --model OPM_SHARED --seed 1101
evaluate --run <run_id> --split test-recombination --locked
intervene --run <run_id> --suite causal-reuse
account --run <run_id>
report --protocol OPM-V1-SPEC
```

Command names may differ, but every operation must be noninteractive and configuration-recorded.

---

## 14. Conformance matrix

| Charter artifact | This specification | Implementation evidence required |
|---|---|---|
| A Typed algebra | Sections 3 | symbolic verifier tests |
| B Generators/renderers | Section 4 | golden render fixtures and property tests |
| C Dataset manifest | Section 5 | frozen JSON manifests and hashes |
| D Tensor contracts | Section 6 | runtime shape/dtype assertions |
| E Primitive boundary | Section 6 | access-control and one-fact tests |
| F Oracle channel | Section 7 | permutation fixture and leakage report |
| G Model configurations | Section 8 | parameter/FLOP accounting report |
| H Losses | Section 9 | equation-to-code tests on fixed tensors |
| I Training protocol | Section 10 | resolved config and event log |
| J Interventions/metrics | Section 11 | intervention golden tests |
| K Statistics | Section 12 | locked analysis script and synthetic validation |
| L Environment | Section 13 | environment manifest and reproduction log |

---

## 15. Blocking dry-run traces

The traces below use illustrative model seed `1101`. The implementation must calculate the actual permutation by ORC-002 and replace symbolic indices `k_LOOKUP`, `k_REVERSE`, `k_CHAIN`, and `k_LIFT` in its generated conformance trace. Numeric neural states cannot be specified before initialization; their exact deterministic equations and shapes are shown instead.

### TRACE-001 Positive one-step `REVERSE`

```yaml
latent:
  entities: [OBJECT:1, OBJECT:2]
  selected_fact: [LINK, 1, 2]
  query: [LINK, 2, 1]
  label: 1
renderings:
  SET:
    fact: paired(e01,e02)
    query: ask:paired(e02,e01)
  SCENE:
    fact: o01 -- o02
    query: ? LINK o02 o01
  PROGRAM:
    fact: connected(v01,v02)
    query: assert? connected(v02,v01)
oracle:
  argument_slots: [2, 1]
  evidence_indices: [0, PAD]
  operation_tokens: [REVERSE, STOP]
  step_count: 1
execution:
  surface_e_0: gather(fact_vectors,0)     # [1,192]
  fact_endpoints_0: [1,2]
  e_0: LayerNorm(surface_e_0 + Wef*Emb([1,2])) # [1,192]
  s_0: MOD-005(query,domain,[2,1],zero_q) # [1,192]
  s_1: C_k_REVERSE(s_0,e_0)              # [1,192]
  logits: decoder(s_1)                    # [1,2]
grouping: [positive, one_step, symmetry, LINK]
```

Structural justification: `LINK` is symmetric, so `LINK(1,2)` entails `LINK(2,1)`.

### TRACE-002 Negative one-step `LOOKUP`

```yaml
latent:
  entities: [OBJECT:1, OBJECT:2, OBJECT:3]
  selected_fact: [BEFORE, 1, 2]
  query: [BEFORE, 1, 3]
  world_constraint: no path from 1 to 3 of length <= 2
  label: 0
renderings:
  SET:
    fact: ordered_before(e01,e02)
    query: ask:ordered_before(e01,e03)
  SCENE:
    fact: o01 <t o02
    query: ? BEFORE o01 o03
  PROGRAM:
    fact: precedes(v01,v02)
    query: assert? precedes(v01,v03)
oracle:
  argument_slots: [1, 3]
  evidence_indices: [0, PAD]
  operation_tokens: [LOOKUP, STOP]
  step_count: 1
execution:
  surface_e_0: gather(fact_vectors,0)     # [1,192]
  fact_endpoints_0: [1,2]
  e_0: LayerNorm(surface_e_0 + Wef*Emb([1,2])) # [1,192]
  s_0: MOD-005(query,domain,[1,3],zero_q) # [1,192]
  s_1: C_k_LOOKUP(s_0,e_0)               # [1,192]
  logits: decoder(s_1)                    # [1,2]
grouping: [negative, one_step, direct, BEFORE, endpoint_corruption]
```

Structural justification: the selected ordered fact does not match the query, and the verifier has excluded an alternate entailment.

### TRACE-003 Positive two-step `CHAIN`

```yaml
latent:
  entities: [OBJECT:1, OBJECT:2, OBJECT:3]
  selected_facts:
    - [BEFORE, 1, 2]
    - [BEFORE, 2, 3]
  query: [BEFORE, 1, 3]
  label: 1
renderings:
  SET:
    facts: ordered_before(e01,e02); ordered_before(e02,e03)
    query: ask:ordered_before(e01,e03)
  SCENE:
    facts: o01 <t o02; o02 <t o03
    query: ? BEFORE o01 o03
  PROGRAM:
    facts: precedes(v01,v02); precedes(v02,v03)
    query: assert? precedes(v01,v03)
oracle:
  argument_slots: [1, 3]
  evidence_indices: [0, 1]
  operation_tokens: [CHAIN, CHAIN]
  step_count: 2
execution:
  surface_e_0: gather(fact_vectors,0)     # [1,192]
  surface_e_1: gather(fact_vectors,1)     # [1,192]
  fact_endpoints: [[1,2],[2,3]]
  e_0: LayerNorm(surface_e_0 + Wef*Emb([1,2])) # [1,192]
  e_1: LayerNorm(surface_e_1 + Wef*Emb([2,3])) # [1,192]
  s_0: MOD-005(query,domain,[1,3],zero_q) # [1,192]
  s_1: C_k_CHAIN(s_0,e_0)                # [1,192]
  s_2: C_k_CHAIN(s_1,e_1)                # [1,192]
  logits: decoder(s_2)                    # [1,2]
grouping: [positive, two_step, transitive, BEFORE]
```

Structural justification: strict `BEFORE` is transitive and the middle entity is exactly `2`.

### TRACE-004 Negative two-step `LIFT`

```yaml
latent:
  entities: [OBJECT:1, CONTAINER:10, CONTAINER:11, CONTAINER:12]
  selected_facts:
    - [DIRECT_IN, 1, 10]
    - [NESTED_IN, 11, 12]
  query: [WITHIN, 1, 12]
  world_constraint: no NESTED_IN path from 10 to 12 of length <= 1
  label: 0
renderings:
  SET:
    facts: member(e01,s10); subset(s11,s12)
    query: ask:within(e01,s12)
  SCENE:
    facts: o01 @ r10; r11 << r12
    query: ? WITHIN o01 r12
  PROGRAM:
    facts: has(c10,v01); imports(c12,c11)
    query: assert? within(v01,c12)
oracle:
  argument_slots: [1, 12]
  evidence_indices: [0, 1]
  operation_tokens: [LIFT, LIFT]
  step_count: 2
execution:
  surface_e_0: gather(fact_vectors,0)      # [1,192]
  surface_e_1: gather(fact_vectors,1)      # [1,192]
  fact_endpoints: [[1,10],[11,12]]
  e_0: LayerNorm(surface_e_0 + Wef*Emb([1,10])) # [1,192]
  e_1: LayerNorm(surface_e_1 + Wef*Emb([11,12])) # [1,192]
  s_0: MOD-005(query,domain,[1,12],zero_q) # [1,192]
  s_1: C_k_LIFT(s_0,e_0)                  # [1,192]
  s_2: C_k_LIFT(s_1,e_1)                  # [1,192]
  logits: decoder(s_2)                     # [1,2]
grouping: [negative, two_step, lift, WITHIN, middle_corruption]
```

Structural justification: the container stored after step one is `10`, but the second fact begins with `11`; the lift cannot compose.

### TRACE-005 Review obligation

The implementation must reproduce these traces as deterministic fixtures, including exact surface token IDs, actual seed-derived module indices, tensor dtypes, and symbolic-verifier outputs. A mismatch blocks `SPEC_APPROVED` or `IMPLEMENTATION_VALIDATION`, depending on when it is discovered.

---

## 16. Known limitations

- Oracle evidence selection removes retrieval from the problem.
- The domains are symbolic renderers, not natural language, pixels, or executable programs.
- Fixed routing does not test learned module discovery.
- Four unused sentinel modules make total parameter efficiency intentionally unfavorable in v1.
- The fact and query encoders may still learn surface normalization; causal tests are required to show primitives contribute specifically.
- Five model seeds provide an initial rather than definitive estimate of optimization variance.
- H4 and qualifier-conditioned exception handling are intentionally absent.
- H2 and H3 require later confirmatory protocols.

These limitations bound the claims; they do not invalidate the Stage A mechanism test.

---

## 17. Approval record

```yaml
approval:
  specification_id: OPM-V1-SPEC
  specification_version: 1.0.0
  approved_commit_or_hash: null
  candidate_normative_prefix_sha256: d29323d511e6b9813a2e874b71e78b2836b2000cebf0d2649abcbe284b626ffe
  approved_by: null
  approval_date: null
  approved_scope: null
  known_exceptions:
    - PyTorch and target accelerator environment remain to be resolved during IMPLEMENTATION_VALIDATION
  next_lifecycle_state: SPEC_APPROVED
```

The specification has passed technical review but awaits an explicit owner statement approving OPM v1 specification version 1.0.0 and its normative digest. Scientific-core implementation and primary experiments remain prohibited.

The candidate normative digest is SHA-256 over the exact UTF-8 bytes from the beginning of this file up to, but excluding, the `## 17. Approval record` heading. This makes the content independently verifiable without creating a self-referential hash. Upon explicit approval, this candidate value is copied into `approved_commit_or_hash` with the `normative-prefix-sha256:` prefix.

---

## 18. Review checklist

- [x] Owner accepts the four relations and inference table.
- [x] Owner accepts `DIRECT_IN`/`NESTED_IN` facts and `WITHIN` query semantics.
- [x] Owner accepts full `BEFORE` reachability and `SAME` equivalence closure for target labels.
- [x] Owner accepts the three renderers as adequate initial domains.
- [ ] Symbolic review confirms every negative is unentailed.
- [x] Primitive one-fact access boundary is accepted.
- [x] Fixed routing and four sentinel modules are accepted.
- [x] Recombination holdout matrix is accepted.
- [x] Dataset sizes and seed namespaces are accepted.
- [x] Tensor dimensions and primitive equations are accepted.
- [x] Canonical selected-fact endpoint bindings are accepted as oracle information.
- [x] Lexer grammar, token IDs, padding, and renderer renaming are specified for implementation validation.
- [x] Canonical loss and training budget are accepted.
- [x] Baseline matching and accounting rules are accepted.
- [x] Domain-generalist architecture and its capacity-advantaged status are accepted.
- [x] Generator helper algorithms and deterministic allocation rules are specified for implementation validation.
- [x] Leakage thresholds are accepted.
- [x] H1 effect threshold and mechanism criteria are accepted.
- [x] H2/H3 scope restrictions are accepted.
- [ ] Environment requirements are feasible in the target workspace.
- [ ] PyTorch version, installation source, lockfile entry, and CPU/CUDA target are approved.
- [x] Four blocking specification dry-run traces have been added.
- [ ] Implementation reproduces and verifies the four traces.
- [ ] Requirement-to-test traceability is complete in the implementation.
- [ ] Approval record is filled by the scientific owner.

Owner confirmation blocks `SPEC_APPROVED`. After approval, the remaining implementation-validation items block `PROTOCOL_FROZEN` and primary runs.


