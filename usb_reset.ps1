<#
Backyard Critter Cam -- USB camera reset (the wedge cure) + its one-time setup.

WHY THIS EXISTS
  The 2026-07 glass-door cam wedges after Modern-Standby suspend cycles: the UVC stream keeps
  delivering frames, but they're torn garbage, and NOTHING app-level fixes it (2026-07-30: 19
  capture reopens + 5 AUTO_WB re-asserts, no effect; a physical unplug/replug fixed it on the
  first try). This script is the software version of that replug: a pnputil disable -> enable
  cycle, which re-enumerates the device.

  pnputil needs admin and the rig runs unprivileged, so the two meet in a scheduled task:
  `-Setup` (run ONCE, elevated -- setup_selfheal.bat does the elevation for you) registers a
  SYSTEM task pointing at a protected copy of this script and grants normal users the right to
  START it. The rig's wedge detector (powerguard.py) then fires `schtasks /run` unattended.

USAGE
  .\usb_reset.ps1 -DryRun     show which device WOULD be cycled (no admin needed) -- run this
                              first if you're unsure the -Pattern matches your camera
  .\usb_reset.ps1 -Setup      one-time task registration (elevated; use setup_selfheal.bat)
  .\usb_reset.ps1             cycle the device now (admin; this is what the task runs)

  -Pattern <regex>  FriendlyName match for the EXTERNAL camera (default 'USB' -- external UVC
                    cams enumerate as e.g. "USB Camera"; built-ins as "Integrated Camera").
                    Must match EXACTLY ONE present device or the script refuses to act.
  Re-run -Setup after changing -Pattern or editing this file (the task runs the copy).
#>
param(
    [string]$Pattern = 'USB',
    [string]$LogPath = '',
    [string]$TaskName = 'BackyardCritterCam-UsbReset',
    [switch]$Setup,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'

if (-not $LogPath) { $LogPath = Join-Path $PSScriptRoot 'logs\usb_reset.log' }

function Write-Log([string]$msg) {
    $line = "{0:yyyy-MM-ddTHH:mm:ss}  {1}" -f (Get-Date), $msg
    Write-Host $line
    try { Add-Content -Path $LogPath -Value $line -Encoding utf8 } catch {}
}

function Get-CameraDevices {
    # Camera-class first; some UVC units enumerate under Image instead, so take both.
    $devs = @()
    try { $devs = @(Get-PnpDevice -Class Camera, Image -ErrorAction SilentlyContinue |
                    Where-Object { $_.Present }) } catch {}
    return $devs
}

function Resolve-TargetDevice {
    $all = Get-CameraDevices
    $hits = @($all | Where-Object { $_.FriendlyName -match $Pattern })
    if ($hits.Count -ne 1) {
        Write-Log "need EXACTLY ONE present Camera/Image device matching -Pattern '$Pattern'; found $($hits.Count)."
        foreach ($d in $all) { Write-Log ("  seen: '{0}'  [{1}]  status={2}" -f $d.FriendlyName, $d.InstanceId, $d.Status) }
        Write-Log "pick a -Pattern that matches only the external cam, then re-run (and re-run -Setup so the task learns it)."
        return $null
    }
    return $hits[0]
}

# ---- One-time setup (elevated) ------------------------------------------------------
if ($Setup) {
    $pr = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $pr.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Setup needs an elevated shell -- run setup_selfheal.bat (it asks for UAC), or start PowerShell as admin.'
    }
    # Fail HERE on a bad pattern, not at 2 AM when the rig actually needs the reset.
    $dev = Resolve-TargetDevice
    if ($null -eq $dev) { throw "no unambiguous camera match for -Pattern '$Pattern' -- see the log lines above." }
    Write-Log ("setup: target device '{0}'  [{1}]" -f $dev.FriendlyName, $dev.InstanceId)

    # The task runs as SYSTEM, so it must NOT execute a user-writable file (that would hand any
    # local process a SYSTEM escalation). Copy this script somewhere only admins can write.
    $destDir = Join-Path $env:ProgramData 'BackyardCritterCam'
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    $dest = Join-Path $destDir 'usb_reset.ps1'
    Copy-Item -Path $PSCommandPath -Destination $dest -Force
    icacls $destDir /inheritance:r /grant 'SYSTEM:(OI)(CI)F' /grant 'Administrators:(OI)(CI)F' /grant 'Users:(OI)(CI)RX' | Out-Null

    # The task logs beside the rig's own logs so one folder holds the whole post-mortem.
    $taskLog = Join-Path $PSScriptRoot 'logs\usb_reset.log'
    $psExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $arg = "-NoProfile -ExecutionPolicy Bypass -File `"$dest`" -Pattern `"$Pattern`" -LogPath `"$taskLog`""
    $action = New-ScheduledTaskAction -Execute $psExe -Argument $arg
    $principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
    # -AllowStartIfOnBatteries is the whole point: wedges HAPPEN on battery evenings.
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal -Settings $settings -Force | Out-Null

    # Grant normal users read+execute on the task, so the UNPRIVILEGED rig can `schtasks /run` it.
    $svc = New-Object -ComObject 'Schedule.Service'
    $svc.Connect()
    $folder = $svc.GetFolder('\')
    $folder.GetTask($TaskName).SetSecurityDescriptor('D:(A;;FA;;;BA)(A;;FA;;;SY)(A;;GRGX;;;AU)', 0)

    Write-Log ("setup: registered task '{0}' -> {1}" -f $TaskName, $dest)
    Write-Log "setup: done. Test it any time (no admin needed) with:  schtasks /run /tn $TaskName"
    exit 0
}

# ---- Dry run (no admin needed) ------------------------------------------------------
if ($DryRun) {
    $dev = Resolve-TargetDevice
    if ($null -eq $dev) { exit 2 }
    Write-Log ("dry run: WOULD cycle '{0}'  [{1}]  status={2}" -f $dev.FriendlyName, $dev.InstanceId, $dev.Status)
    exit 0
}

# ---- The actual cycle (admin; the scheduled task lands here) ------------------------
$dev = Resolve-TargetDevice
if ($null -eq $dev) { exit 2 }
$id = $dev.InstanceId
Write-Log ("cycling '{0}'  [{1}] -- disable, pause, enable (the software replug)." -f $dev.FriendlyName, $id)

pnputil /disable-device "$id" | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Log "disable FAILED (pnputil rc=$LASTEXITCODE). Not elevated? The rig triggers this via the scheduled task for exactly that reason."
    exit 1
}
Start-Sleep -Seconds 3

# Enable can transiently fail while the stack is still tearing down -- retry a few times, and
# NEVER walk away leaving the camera disabled (that would be worse than the wedge).
$enabled = $false
foreach ($try in 1..3) {
    pnputil /enable-device "$id" | Out-Null
    if ($LASTEXITCODE -eq 0) { $enabled = $true; break }
    Write-Log "enable attempt $try failed (pnputil rc=$LASTEXITCODE); retrying in 2 s."
    Start-Sleep -Seconds 2
}
if (-not $enabled) {
    Write-Log "enable FAILED after 3 attempts -- the camera may be left DISABLED. Re-enable it in Device Manager (or replug the cable, which also re-enumerates)."
    exit 1
}
Start-Sleep -Seconds 2
$after = $null
try { $after = (Get-PnpDevice -InstanceId $id -ErrorAction SilentlyContinue).Status } catch {}
Write-Log ("cycle complete -- device status now '{0}'. The rig's reconnect loop will pick it back up." -f $after)
exit 0
