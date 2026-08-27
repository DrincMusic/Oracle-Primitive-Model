# Agent setup guide

This guide gives coding agents one unambiguous entry point while keeping three different goals from
being confused with one another.

## Tell an agent what outcome you want

For closed-result verification, use this prompt:

> Set up this repository for OPM v1.1.4 closed-result verification. Read `AGENTS.md` and
> `OPM_EXPERIMENT.json`, run the agent doctor and verifier, do not edit the frozen workspace, and
> report the closeout freeze hash, Merkle root, and pass/fail status.

For a development environment and source tests, use this prompt:

> Set up the standalone OPM test environment. Read `AGENTS.md`, run `python tools/opm_agent.py doctor`,
> then run `python tools/opm_agent.py setup`. Do not modify `studies/v1.1.4/workspace`. Report the
> dependency or platform blocker if setup cannot complete, otherwise report the test count.

For full retraining, ask the agent to assess readiness rather than launch immediately:

> Assess this repository for a fresh full OPM v1.1.4 computational reproduction. Do not launch
> training. Read the external-evidence manifest and frozen execution records, identify the missing
> reproduction-bundle components, estimate storage and compute, and propose a new output workspace
> that cannot alter the historical snapshot.

## Entry-point commands

The doctor is read-only and uses only Python's standard library:

```text
python tools/opm_agent.py doctor
```

Closed-result verification is also dependency-light and does not download anything:

```text
python tools/opm_agent.py verify
```

Development setup creates `.venv`, downloads the dependencies in `requirements-study.txt`, verifies
the closed result, and runs the standalone source tests:

```text
python tools/opm_agent.py setup
```

On Linux and Windows, setup defaults to the smaller PyTorch 2.8 CPU wheel used by CI. This is enough
for verification and source tests. Use `--torch-profile default` only when the platform-default wheel
is intentional; GPU training requires a separately planned CUDA environment and is not launched by
this setup command.

An agent can inspect the exact setup commands without changing the machine:

```text
python tools/opm_agent.py setup --dry-run
```

Python 3.12 is the tested version. Python 3.11 is the minimum accepted by the onboarding tool.

## Windows checkout

The byte-preserving historical tree contains deep paths. Use a short destination and Git long-path
support:

```powershell
git -c core.longpaths=true clone https://github.com/DrincMusic/Oracle-Primitive-Model.git C:\opm
cd C:\opm
python tools\opm_agent.py doctor
```

## What is and is not reproducible now

The Git checkout can reproduce the publication inventory, closeout hashes, claim count, final-report
identity, and the 96-test source baseline. It cannot yet perform a clean full computational rerun from
a single command. The persistent external archive is pending, and the frozen execution workspace is
a completed historical record rather than a fresh-run template.

Large artifacts must never be committed directly to Git. When the external archive and fresh-run
runbook are ready, they should be connected through content hashes and new output directories, leaving
`studies/v1.1.4/workspace` untouched.
