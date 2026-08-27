# OPM v1.1.4 primary-execution handoff

> **CURRENT LIVE OPERATIONAL HANDOFF. Read this file before any historical OPM execution handoff.**
>
> `OPM_V1_1_4_EXECUTION_HANDOFF.md` is historical pilot provenance. Never recover or launch a pilot
> from its active-PID snapshot, checkpoint path, early matrix table, or pilot-era required actions.
>
> The current-state block below must be replaced in place whenever a primary row starts, is
> interrupted, or completes. Do not leave stale state at the beginning and append a correction later.

## Current authoritative state

```yaml
state_recorded_on: 2026-08-27
lifecycle: PRIMARY_RUNS
execution_status: COMPLETED
protocol_frozen: true
canonical_training_authorized: true
pilot_matrix_state: COMPLETED
pilot_rows_total: 24
pilot_rows_completed: 24
selection_state: FROZEN
selected_learning_rate: 0.0006
selected_dropout: 0.0
primary_matrix_state: COMPLETED
primary_rows_total: 20
primary_rows_pending: 0
primary_rows_started: 20
primary_rows_completed: 20
active_primary_run: null
active_primary_condition: null
active_primary_model_seed: null
active_primary_process_id: null
active_primary_worker_child_process_id: null
active_monitor_process_id: null
active_monitor_worker_child_process_id: null
primary_live_state: C:/Users/Dwight/Documents/ChatGPT/RLMGraph/evidence/primary_runs/v1.1.4/primary-live-state.json
last_completed_primary_run: 2cfd5073e651505a564c643d
last_completed_primary_condition: DOMAIN_GENERALIST
last_completed_primary_model_seed: 5505
latest_published_event_step: 50000
latest_verified_checkpoint_step: 50000
latest_verified_checkpoint_sha256: 17887001c80359da9f98b740c6d9573e1b6c8c55e3727074e9ad29a49117f302
latest_verified_macro_validation_accuracy: 0.9976388888888889
first_pending_condition: null
first_pending_model_seed: null
first_pending_expected_run_id: null
approved_opm_source_sha256: c885ef2eb39feaa5a1ab2116b5bc387fff4bf6713b4b1292befaf084787d6366
sealed_label_access_authorized: false
aggregate_test_evaluation_authorized: false
claim_decisions_authorized: false
sealed_labels_accessed: false
aggregate_test_evaluation_performed: false
claim_decisions_performed: false
```

All 20 primary rows completed and passed full reconciliation by 2026-08-27: exactly 50,000 contiguous
events each, 100 checkpoint events/files each, zero checkpoint hash mismatches, matching selected
checkpoints, matching manifest/summary identities, and `COMPLETED` ledger rows. No primary trainer or
monitor is active, and no row is pending. No additional primary launch is authorized by this completed
matrix.

## Authority and immutable inputs

| Artifact | Current SHA-256 |
|---|---|
| Status | `1c1ad05b7b36dfa7bbbc5f74f1b689b21c2ef795b8f314a9b7b41f47637b32bc` |
| Historical pilot handoff with supersession warning | `ed7febd649c7df1075f005c6b6747f08f5a5c0b5c0c44dc3b2fddc3c5da9fa01` |
| v1.1.4 transition JSON | `ec597e19bfc299a5c890bcddd46e4daf95fcb034081d974063eeb0889385b5c4` |
| Effective v1.1.4 erratum | `8108a5b0be1246429c1ef20df02189886eef78f24b764e032f2962afea13b145` |
| Completed pilot matrix | `ad90a1097bbfe4e950f8db2a123ae742fb0eb400b8d0dd72f4d07a419783d5be` |
| Completed primary matrix | `79107e31d9faee4d8f42685f801b63b503970dace55879de3c087f506338dc82` |
| Declared-seed primary executor | `58687867494f3bac5c48675c104fd89ad9f69118fdefbade63aa551af929573a` |
| Atomic live-state monitor | `962d02f0aa23cf5b18408622d284bb9fad7f72964eafe65b8e8871d959a782a7` |
| Canonical train JSONL | `4f2c07bfc0400c992a936fa7b64f3fabefcd2f8b29064bf610a7c83c283deafa` |
| Canonical validation JSONL | `1dee006f0db36bf77925f6f375a8d776f4049059c2994d65ee1f1e9eed58236f` |

Authoritative paths:

```yaml
status: C:/Users/Dwight/Documents/ChatGPT/RLMGraph/evidence/implementation_validation/OPM_V1_STATUS.md
transition: C:/Users/Dwight/Documents/ChatGPT/RLMGraph/evidence/implementation_validation/OPM_V1_1_4_TRAINING_EXECUTION_TRANSITION.json
erratum: C:/Users/Dwight/Documents/ChatGPT/RLMGraph/OPM_V1_1_4_TRAINING_EXECUTION_ERRATUM.md
pilot_matrix: C:/Users/Dwight/Documents/ChatGPT/RLMGraph/evidence/primary_runs/v1.1.4/pilot-matrix.json
primary_matrix: C:/Users/Dwight/Documents/ChatGPT/RLMGraph/evidence/primary_runs/v1.1.4/primary-matrix.json
primary_executor: C:/Users/Dwight/Documents/ChatGPT/RLMGraph/scripts/opm_primary_executor.py
live_state_monitor: C:/Users/Dwight/Documents/ChatGPT/RLMGraph/scripts/opm_live_state_monitor.py
primary_artifact_root: C:/Users/Dwight/Documents/ChatGPT/RLMGraph/evidence/primary_runs/v1.1.4/primary
primary_live_state: C:/Users/Dwight/Documents/ChatGPT/RLMGraph/evidence/primary_runs/v1.1.4/primary-live-state.json
train: C:/Users/Dwight/Documents/ChatGPT/RLMGraph/evidence/implementation_validation/generated/v1.1.3-canonical-data/train.v1.1.3.canonical.jsonl
validation: C:/Users/Dwight/Documents/ChatGPT/RLMGraph/evidence/implementation_validation/generated/v1.1.3-canonical-data/validation.v1.1.3.canonical.jsonl
```

The executor is external to `src/rlmgraph/opm` and pins the pilot-proven source revision. Do not edit
the approved OPM source tree, pass an alternate revision, supply a different seed, or supply training
hyperparameters. The executor recalculates the pilot selection, verifies all 24 pilot summaries,
verifies canonical input hashes, derives the frozen pair, validates all 20 primary identities, and
fails closed on duplicates, concurrent rows, unreconciled failures, or unsafe recovery boundaries.

The four seed-1101 primary run IDs equal the corresponding selected-configuration pilot IDs because
identity is content-addressed by source, data, configuration, condition, and seed. This does not make
pilot artifacts primary replicates. Only directories beneath `primary/` and rows in
`primary-matrix.json` may enter later primary-result accounting; `pilots/` remains excluded.

## Mandatory cold-resume read order

Read, in order:

1. This file.
2. `OPM_V1_STATUS.md`.
3. `OPM_V1_1_4_TRAINING_EXECUTION_TRANSITION.json`.
4. `pilot-matrix.json` and its frozen `selection` object.
5. `primary-matrix.json`.
6. `scripts/opm_primary_executor.py` and `scripts/opm_live_state_monitor.py`.
7. Only when a primary row is `RUNNING` or `INTERRUPTED`: that row's `run-manifest.json`,
   `events.jsonl`, summary if present, and checkpoint directory.

Do not open any file beneath a `sealed-labels` directory. Do not use the pilot-era `live-state.json`
as primary state. Primary monitoring writes only `primary-live-state.json`.

## Required next actions

1. Verify the status authority fields, approved source digest, completed pilot-ledger hash, and current
   completed primary-matrix hash against this file before any later lifecycle work.
2. Preserve the completed matrix, manifests, summaries, event streams, checkpoints, status, and this
   handoff as the authoritative Stage 1 primary-training record.
3. Do not launch or recover another primary row: all 20 declared rows are complete and the matrix has
   no pending or interrupted entry.
4. Require a separately recorded transition and applicable authority before beginning any subsequent
   lifecycle operation; do not infer that transition from primary completion alone.
5. Never access sealed labels, perform aggregate test evaluation, or make claim decisions unless a
   separate artifact explicitly authorizes that action.

## Disabled primary-launch template

The matrix is complete and no primary row remains pending. The template below is retained only as
execution provenance. Do not execute it without new explicit authority and a new matrix row.

From the repository root, perform a write-free preflight:

```powershell
$python = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$executor = (Resolve-Path ".\scripts\opm_primary_executor.py").Path
$train = (Resolve-Path ".\evidence\implementation_validation\generated\v1.1.3-canonical-data\train.v1.1.3.canonical.jsonl").Path
$validation = (Resolve-Path ".\evidence\implementation_validation\generated\v1.1.3-canonical-data\validation.v1.1.3.canonical.jsonl").Path
$pilotMatrix = (Resolve-Path ".\evidence\primary_runs\v1.1.4\pilot-matrix.json").Path
$condition = "DOMAIN_GENERALIST"
$seed = 5505
$preflightText = & $python $executor $condition $seed $train $validation $pilotMatrix --preflight-only | Out-String
if ($LASTEXITCODE -ne 0) { throw "Primary preflight failed." }
$preflight = $preflightText | ConvertFrom-Json
if ($preflight.state -ne "PASS") { throw "Primary preflight did not return PASS." }
```

Before launch, independently confirm no matching process exists. Process absence alone is not enough;
the ledger and artifact root must also agree:

```powershell
Get-CimInstance Win32_Process |
  Where-Object {
    $_.Name -match '^(opm-v1|python)' -and
    $_.CommandLine -match '(opm_primary_executor|primary-pilot-run|primary-run|rlmgraph\\opm)'
  } |
  Select-Object ProcessId, Name, CommandLine
```

Launch exactly one hidden executor process and preserve stdout/stderr:

```powershell
$root = (Resolve-Path ".").Path
$logs = Join-Path $root "evidence\primary_runs\v1.1.4\logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$stamp = Get-Date -Format "yyyyMMddTHHmmss"
$conditionSlug = $condition.ToLowerInvariant()
$stdout = Join-Path $logs "primary-$conditionSlug-seed$seed-$stamp.stdout.log"
$stderr = Join-Path $logs "primary-$conditionSlug-seed$seed-$stamp.stderr.log"
$arguments = @(
  $executor, $condition, [string]$seed, $train, $validation, $pilotMatrix,
  "--device", "cuda:0"
)
$trainer = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $root `
  -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
```

Wait only for the immutable run manifest to appear; stop if the process exits or the bounded wait
expires:

```powershell
$runId = [string]$preflight.run_id
$runRoot = Join-Path $root "evidence\primary_runs\v1.1.4\primary"
$manifest = Join-Path $runRoot "$runId\run-manifest.json"
$deadline = (Get-Date).AddSeconds(60)
while (-not (Test-Path -LiteralPath $manifest)) {
  if ($trainer.HasExited) { throw "Primary executor exited before writing its manifest." }
  if ((Get-Date) -ge $deadline) { throw "Timed out waiting for the primary manifest." }
  Start-Sleep -Seconds 1
}
```

Validate the manifest and `RUNNING` ledger transition, then start the hidden monitor:

```powershell
$sourceDigest = "c885ef2eb39feaa5a1ab2116b5bc387fff4bf6713b4b1292befaf084787d6366"
$monitorScript = (Resolve-Path ".\scripts\opm_live_state_monitor.py").Path
$liveState = Join-Path $root "evidence\primary_runs\v1.1.4\primary-live-state.json"
$monitorStdout = Join-Path $logs "primary-monitor-$conditionSlug-seed$seed-$stamp.stdout.log"
$monitorStderr = Join-Path $logs "primary-monitor-$conditionSlug-seed$seed-$stamp.stderr.log"
$monitorArguments = @(
  $monitorScript,
  "--run-root", $runRoot,
  "--run-id", $runId,
  "--condition", $condition,
  "--learning-rate", "0.0006",
  "--dropout", "0.0",
  "--seed", [string]$seed,
  "--pid", [string]$trainer.Id,
  "--source-digest", $sourceDigest,
  "--output", $liveState,
  "--poll-seconds", "5"
)
$monitor = Start-Process -FilePath $python -ArgumentList $monitorArguments `
  -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $monitorStdout `
  -RedirectStandardError $monitorStderr -PassThru
```

Immediately replace the opening current-state block with the new matrix state, active run ID, trainer
PID, monitor PID, live-state path, and updated primary-matrix hash. Update the status launch snapshot
at the same time.

## Continuous monitoring and user updates

`primary-live-state.json` is atomically replaced after each stable checkpoint and at terminal state.
Read it without stopping either process:

```powershell
Get-Content -LiteralPath evidence\primary_runs\v1.1.4\primary-live-state.json -Raw
Get-Process -Id $trainer.Id,$monitor.Id -ErrorAction SilentlyContinue |
  Select-Object Id, ProcessName, StartTime
```

Keep the external monitor alive for the entire run. Provide chat updates from the state file and the
latest complete validation event. A chat response is not a reason to kill, interrupt, or abandon the
monitor. If the user asks to keep monitoring, continue until completion or a genuine interruption.

The state file's three safety fields describe monitor activity only. Global restrictions remain
governed by the lifecycle transition and audit artifacts.

## Recovery boundary

If power loss, process death, or another infrastructure interruption occurs:

1. Do not start another row or silently restart this row.
2. Preserve the existing run directory, logs, manifest, event log, and checkpoints.
3. Verify the manifest identity and canonical fingerprints.
4. Find the newest complete stable 500-step checkpoint and verify its SHA-256 against its event.
5. Confirm the event log is contiguous and ends exactly at that checkpoint. If it contains a corrupt
   tail or later uncheckpointed events, preserve the original byte-for-byte with its SHA-256 and record
   an explicit recovery entry before reconciling; never silently truncate or overwrite evidence.
6. Resume only the identical condition, seed, run ID, code revision, and artifact root by adding
   `--resume-checkpoint <verified checkpoint>` to the executor command.
7. Start a fresh external monitor for the resumed process, then replace the opening current state and
   update status.

The executor accepts recovery only when the matrix row is `RUNNING` or `INTERRUPTED`, the checkpoint
is inside the expected content-addressed primary directory, its step is a canonical boundary below
50,000, the manifest matches exactly, the event sequence is contiguous, the log ends at the selected
checkpoint, and the event/checkpoint hashes agree.

## Completion verification and advancement gate

Do not advance merely because the process exits or `summary.json` exists. Verify:

- the row's manifest matches condition, seed, frozen configuration, approved source revision, and
  canonical train/validation fingerprints;
- `summary.json` reports 50,000 completed steps and the requested run identity;
- `events.jsonl` contains exactly 50,000 nonblank, contiguous step records;
- exactly 100 checkpoint events and 100 checkpoint files exist;
- every checkpoint file SHA-256 matches its event record;
- the selected checkpoint exists and matches the summary SHA-256;
- the executor atomically marked exactly that primary-matrix row `COMPLETED` with the same summary;
- no sealed-label, aggregate-evaluation, or claim-decision authority or activity changed.

Then replace the opening current-state block and hash table, update `OPM_V1_STATUS.md`, and only then
read the next first `PENDING` row from the primary matrix. If any verification fails, set no new row in
motion and preserve the evidence for reconciliation.

## Validation record

```yaml
executor_validation_recorded_on: 2026-08-24
ruff: PASS
opm_tests: 62 passed
canonical_preflight: PASS
canonical_preflight_condition: OPM_SHARED
canonical_preflight_seed: 1101
canonical_preflight_run_id: 3983ca0fcc3eafacd61ffc44
canonical_preflight_created_artifacts: false
canonical_optimizer_steps_started: false
```
