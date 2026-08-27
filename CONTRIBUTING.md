# Contributing

The `studies/v1.1.4/workspace` tree is an immutable historical snapshot. Do not reformat, rename,
modernize, or repair files inside it. Even harmless-looking edits invalidate bound hashes and can make
the closeout verifier fail.

Contributions may improve root documentation, verification tooling, packaging, or future OPM
versions. Changes should:

1. Keep v1.1.4 byte-identical.
2. Add or update tests for behavioral changes.
3. Run `python tools/verify_public_export.py --repository .`.
4. State whether the change affects the frozen study, future software, or documentation only.

For a complete local setup and test pass, run `python tools/opm_agent.py setup`. Agents should read
`AGENTS.md` before making changes.

Scientific reinterpretations should cite the frozen report and clearly distinguish new analysis from
the preregistered v1.1.4 decisions.
