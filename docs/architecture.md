# Architecture

OPM tests whether a shared bank of primitive transformations supports systematic recombination across
multiple rendered domains. The v1.1.4 implementation compares the shared model with untied,
procedure-clone, and domain-generalist controls under frozen data, training, and evaluation rules.

The historical implementation remains under `studies/v1.1.4/workspace/src/rlmgraph/opm`. That path
is preserved for integrity; it is not a recommendation for the namespace of future versions.

The scientific workflow is separated into independently authorized stages: protocol and training,
label-blind post-primary generation, sealed aggregation, claim decision, and read-only study closeout.
Each later stage consumes frozen outputs from the prior stage and records exact hashes.
