# Reproducibility

## Closed-result verification

Closed-result verification requires only Python's standard library and the files committed to this
repository:

```powershell
python tools/verify_public_export.py --repository .
```

This checks the public export inventory and invokes the original v1.1.4 closeout verifier. A passing
result establishes byte identity and internal consistency of the published closeout package; it does
not rerun model training or regenerate the large Stage-1 evidence.

## Source tests

Install the study dependencies, then run the OPM-scoped tests from the historical workspace:

```powershell
python -m pip install -r requirements-study.txt
cd studies/v1.1.4/workspace
$env:PYTHONPATH = "src;."
python -m pytest tests -q
```

## Full computational reproduction

Full reproduction additionally requires compute, regenerated canonical datasets, or the externally
archived checkpoints and row-level artifacts. Their expected paths, sizes, and SHA-256 identities are
recorded in `OPM_EXTERNAL_EVIDENCE_MANIFEST.json`. The external archival release is not yet complete.
