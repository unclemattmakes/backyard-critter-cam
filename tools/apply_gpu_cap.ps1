<#
    apply_gpu_cap.ps1 -- cap the GPU's board power, now and at every boot. RUN AS ADMINISTRATOR.

WHY
---
The host hangs under sustained load with no bugcheck, no dump and no WHEA record (see
docs/host-instability-2026-08.md). The leading hypothesis is power delivery: an ~8-year-old
X299 build whose PSU was specified years before the RTX 30-series, whose microsecond transients
run far above the card's nameplate draw. Capping board power flattens those transients. If the
rig then survives a night it could not survive uncapped, the fault is power, and that is worth
far more than any amount of further log-reading.

140 W is deliberately well under the 270 W default -- this is a diagnostic, not a tuning pass.
The rig barely notices: MegaDetector runs on a motion gate, so the GPU is idle most of the time
and the difference on the frames that do run is milliseconds.

WHY IT NEEDS A BOOT TASK
------------------------
`nvidia-smi -pl` does not persist. NVIDIA's persistence mode is not supported on Windows, so the
limit is gone the moment the machine restarts -- which, on a box being investigated for crashing,
is precisely when it will be lost without anyone noticing. The registered task reapplies it as
SYSTEM at every startup, with retries, because at boot the task can easily beat the driver.

USAGE (needs admin, but elevates itself -- just approve the UAC prompt)
    powershell -ExecutionPolicy Bypass -File tools\apply_gpu_cap.ps1            # apply 140 W + persist
    powershell -ExecutionPolicy Bypass -File tools\apply_gpu_cap.ps1 -Watts 180 # a different cap
    powershell -ExecutionPolicy Bypass -File tools\apply_gpu_cap.ps1 -Remove    # undo: restore default, drop the task
#>
[CmdletBinding()]
param(
    [int]$Watts = 140,
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'
$TASK = 'Backyard critter-cam GPU power cap'

# Self-elevate rather than refuse. The first version of this script exited with an error when run
# unelevated -- which is INVISIBLE if you started it by double-clicking or "Run with PowerShell",
# because the window closes on the way out. It looks like it worked: the GPU is still capped from
# whenever `nvidia-smi -pl` was last run by hand, so only the boot task is missing, and you find
# out at the next reboot. Re-launch through UAC instead, and leave the new window open (-NoExit)
# so whoever ran it can actually read the outcome.
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Not elevated -- re-launching through UAC. Approve the prompt to continue."
    $argList = @('-NoExit', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"", '-Watts', $Watts)
    if ($Remove) { $argList += '-Remove' }
    try {
        Start-Process powershell.exe -Verb RunAs -ArgumentList $argList
    } catch {
        Write-Warning "UAC was declined, so nothing has changed. Re-run and approve the prompt, or start an elevated PowerShell yourself."
        exit 1
    }
    exit 0
}

$smi = (Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source
if (-not $smi) { $smi = 'C:\Windows\System32\nvidia-smi.exe' }
if (-not (Test-Path $smi)) { Write-Error "nvidia-smi not found."; exit 1 }

if ($Remove) {
    & $smi -pm 0 2>&1 | Out-Null
    $default = (& $smi --query-gpu=power.default_limit --format=csv,noheader,nounits) | Select-Object -First 1
    if ($default) {
        & $smi -pl ([int][double]$default.Trim()) | Out-Null
        Write-Host "Restored the default power limit ($([int][double]$default.Trim()) W)."
    }
    Unregister-ScheduledTask -TaskName $TASK -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed the boot task '$TASK'."
    exit 0
}

# --- apply now -------------------------------------------------------------------------------
& $smi -pl $Watts
if ($LASTEXITCODE -ne 0) { Write-Error "Failed to set the power limit (exit $LASTEXITCODE)."; exit 1 }

# --- and at every boot -----------------------------------------------------------------------
# -Retry: at startup this task regularly beats the display driver, and a single attempt would
# silently leave the card uncapped for the whole session.
$inner = "for (`$i=0; `$i -lt 30; `$i++) { & '$smi' -pl $Watts; if (`$LASTEXITCODE -eq 0) { break }; Start-Sleep -Seconds 10 }"
$action    = New-ScheduledTaskAction -Execute 'powershell.exe' `
                -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -Command `"$inner`""
$trigger   = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
                -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -MultipleInstances IgnoreNew -StartWhenAvailable

Register-ScheduledTask -TaskName $TASK -Action $action -Trigger $trigger -Principal $principal `
    -Settings $settings -Description 'Reapplies the diagnostic GPU board-power cap at boot; nvidia-smi -pl does not persist on Windows.' -Force | Out-Null

$now = (& $smi --query-gpu=enforced.power.limit --format=csv,noheader) | Select-Object -First 1
Write-Host "GPU power limit is now: $now"
Write-Host "Registered '$TASK' to reapply it at every startup."
Write-Host "Undo with: powershell -ExecutionPolicy Bypass -File tools\apply_gpu_cap.ps1 -Remove"
