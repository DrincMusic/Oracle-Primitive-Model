# OPM v1.1.4 frozen study

The `workspace` directory preserves the historical paths and bytes needed by the original closeout
verifier. Run commands from the repository root unless a command explicitly changes into the study
workspace.

```powershell
python tools/verify_public_export.py --repository .
```

Do not edit files under `workspace`. New documentation or software belongs outside the frozen tree or
in a later version.
