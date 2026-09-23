<#
    hosthealth.ps1 -- a black-box flight recorder for the machine the rig runs on.

WHY
---
Between 2026-08-23 and 2026-08-24 the rig's host (OLVR) hard-hung twice while the rig was
running, and it has hung the same way 16 times since 2025-10. Every one of those left NOTHING
behind: no bugcheck, no crash dump (dumps ARE enabled -- CrashDumpEnabled=7), no WHEA record,
no TDR. That is the signature of a CPU that stops retiring instructions rather than a driver
fault, and it means the usual post-mortem sources are all empty by definition.

So the evidence has to be collected BEFORE the hang. This script samples the host every few
seconds and appends one CSV row per sample. After the next freeze, the last row on disk is the
last moment the machine was alive -- temperature, GPU power draw, CPU load and disk queue
included. That turns "it froze at some point overnight" into a timestamped approach to the edge.

WRITE-THROUGH, NOT BUFFERED
---------------------------
A hard hang loses everything sitting in the OS write cache, which is exactly the last few
seconds we care about most. The file is therefore opened with FileOptions::WriteThrough and
flushed on every sample, so each row is on the device before the next one is taken. It costs a
few hundred microseconds per row and it is the entire point of the script.

WHAT IT CANNOT SEE
------------------
CPU package and VRM temperatures are not exposed to userspace on this board (no ACPI thermal
zones), and those are central to the thermal hypothesis. HWiNFO64 in sensors-only mode can log
them to its own CSV and is the right companion to this script -- run both.

USAGE
    powershell -File tools\hosthealth.ps1                 # 5s samples, until stopped
    powershell -File tools\hosthealth.ps1 -Interval 2     # finer grain
    powershell -File tools\hosthealth.ps1 -Once           # one sample, print it, exit (a smoke test)

Registered as the scheduled task "Backyard critter-cam host telemetry", which starts it at logon.
#>
[CmdletBinding()]
param(
    [int]$Interval = 5,
    [string]$OutDir,
    [int]$KeepDays = 30,
    [switch]$Once
)

$ErrorActionPreference = 'Continue'

if (-not $OutDir) { $OutDir = Join-Path (Split-Path $PSScriptRoot -Parent) 'logs\hosthealth' }
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }

$COLUMNS = 'ts,uptime_s,gpu_temp_c,gpu_power_w,gpu_util_pct,gpu_mem_mb,gpu_sm_mhz,gpu_pstate,gpu_fan_pct,gpu_pcie_gen,cpu_pct,mem_avail_mb,disk_queue,rig_procs,rig_rss_mb'

# --- perf counters are created ONCE; rebuilding them per sample would make the recorder itself
# --- a meaningful load, which would corrupt the very measurement it exists to take.
function New-Counter($category, $counter, $instance) {
    try {
        $c = if ($instance) { New-Object System.Diagnostics.PerformanceCounter $category, $counter, $instance }
             else           { New-Object System.Diagnostics.PerformanceCounter $category, $counter }
        $null = $c.NextValue()   # first read is always 0 -- prime it
        return $c
    } catch { return $null }
}
$cpuCounter  = New-Counter 'Processor' '% Processor Time' '_Total'
$memCounter  = New-Counter 'Memory' 'Available MBytes' $null
$diskCounter = New-Counter 'PhysicalDisk' 'Avg. Disk Queue Length' '_Total'

function Read-Counter($c) {
    if (-not $c) { return '' }
    try { return [math]::Round($c.NextValue(), 1) } catch { return '' }
}

function Get-GpuSample {
    # One nvidia-smi call for every GPU field; spawning it once per sample is cheap next to
    # spawning it eight times.
    try {
        $q = 'temperature.gpu,power.draw,utilization.gpu,memory.used,clocks.current.sm,pstate,fan.speed,pcie.link.gen.current'
        $raw = & nvidia-smi --query-gpu=$q --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $raw) { return @('', '', '', '', '', '', '', '') }
        $parts = ($raw | Select-Object -First 1).Split(',') | ForEach-Object { $_.Trim() }
        # '[N/A]' is what a zero-RPM fan and a few other fields report; blank reads better in a chart.
        return $parts | ForEach-Object { if ($_ -match '^\[?N/?A\]?$') { '' } else { $_ } }
    } catch { return @('', '', '', '', '', '', '', '') }
}

function Get-RigSample {
    # Counted by name, not command line: Get-Process is in-process and costs nothing, where a
    # Win32_Process query with a CommandLine filter is a WMI round-trip every few seconds.
    # We only need "was the rig up, and how big was it" to correlate against a freeze.
    try {
        $p = @(Get-Process -Name python, pythonw, ffmpeg -ErrorAction SilentlyContinue)
        if (-not $p) { return @(0, 0) }
        return @($p.Count, [math]::Round((($p | Measure-Object WorkingSet64 -Sum).Sum) / 1MB, 0))
    } catch { return @('', '') }
}

function Get-BootTime {
    try { return (Get-CimInstance Win32_OperatingSystem).LastBootUpTime } catch { return $null }
}

function New-Writer([string]$path) {
    $isNew = -not (Test-Path $path)
    # WriteThrough is the whole reason this survives a hang -- see the header.
    $fs = New-Object System.IO.FileStream($path, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write,
                                          [System.IO.FileShare]::Read, 4096, [System.IO.FileOptions]::WriteThrough)
    $sw = New-Object System.IO.StreamWriter($fs)
    $sw.AutoFlush = $true
    if ($isNew) { $sw.WriteLine($COLUMNS) }
    return $sw
}

function Get-LastSampleTime([string]$dir) {
    # The last row of the previous file is the last moment the host was alive. Recording it in
    # the new file's session banner is what makes a freeze gap self-evident on inspection.
    try {
        $prev = Get-ChildItem $dir -Filter 'hosthealth-*.csv' -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if (-not $prev) { return $null }
        $last = Get-Content $prev.FullName -Tail 1 -ErrorAction SilentlyContinue
        if ($last -and $last -notmatch '^#' -and $last -notmatch '^ts,') { return $last.Split(',')[0] }
        return $null
    } catch { return $null }
}

# ---------------------------------------------------------------- one-shot smoke test
if ($Once) {
    $g = Get-GpuSample; $r = Get-RigSample
    $row = @((Get-Date -Format 'o'), '', $g[0], $g[1], $g[2], $g[3], $g[4], $g[5], $g[6], $g[7],
             (Read-Counter $cpuCounter), (Read-Counter $memCounter), (Read-Counter $diskCounter), $r[0], $r[1]) -join ','
    Write-Output $COLUMNS
    Write-Output $row
    exit 0
}

# ---------------------------------------------------------------- main loop
$boot = Get-BootTime
$lastSeen = Get-LastSampleTime $OutDir
$day = (Get-Date).ToString('yyyy-MM-dd')
$writer = New-Writer (Join-Path $OutDir "hosthealth-$day.csv")
$writer.WriteLine(("# session start {0} | boot {1} | interval {2}s | last sample before this session: {3}" -f `
    (Get-Date -Format 'o'), $(if ($boot) { $boot.ToString('o') } else { 'unknown' }), $Interval,
    $(if ($lastSeen) { $lastSeen } else { 'none' })))

# Prune old files so an unattended recorder cannot fill the disk it is monitoring.
try {
    Get-ChildItem $OutDir -Filter 'hosthealth-*.csv' -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-$KeepDays) } |
        Remove-Item -Force -ErrorAction SilentlyContinue
} catch { }

try {
    while ($true) {
        try {
            $now = Get-Date
            $today = $now.ToString('yyyy-MM-dd')
            if ($today -ne $day) {
                $writer.Dispose()
                $day = $today
                $writer = New-Writer (Join-Path $OutDir "hosthealth-$day.csv")
            }

            $g = Get-GpuSample
            $r = Get-RigSample
            $up = if ($boot) { [math]::Round(($now - $boot).TotalSeconds, 0) } else { '' }

            $writer.WriteLine((@($now.ToString('o'), $up,
                                 $g[0], $g[1], $g[2], $g[3], $g[4], $g[5], $g[6], $g[7],
                                 (Read-Counter $cpuCounter), (Read-Counter $memCounter),
                                 (Read-Counter $diskCounter), $r[0], $r[1]) -join ','))
        } catch {
            # A recorder that dies on one bad sample is worse than useless -- it looks like a
            # freeze. Swallow, and take the next sample.
            try { $writer.WriteLine("# sample error {0}: {1}" -f (Get-Date -Format 'o'), $_.Exception.Message) } catch { }
        }
        Start-Sleep -Seconds $Interval
    }
} finally {
    try { $writer.WriteLine("# session end {0} (clean stop)" -f (Get-Date -Format 'o')) } catch { }
    try { $writer.Dispose() } catch { }
}
