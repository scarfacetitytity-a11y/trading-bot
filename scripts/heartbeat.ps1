# AiDEN heartbeat — runs every 5 minutes via Task Scheduler.
# Two checks:
#   1. Process alive  — orchestrator is in the process table
#   2. Log freshness  — orchestrator wrote to its log within the last 10 minutes
# If either fails: kill stale process (if any), restart clean.
# Also ensures watchdog.ps1 is running — restarts it if dead.
#
# Register with: powershell -ExecutionPolicy Bypass -File scripts\register_heartbeat.ps1

param(
    [string]$BotDir     = "C:\aiden-bot",
    [int]   $StaleMinutes = 10
)

$PythonCmd = "$BotDir\venv\Scripts\python.exe"
$LogFile   = "$BotDir\logs\bot.log"
$HBLog     = "$BotDir\logs\heartbeat.log"

function Write-HB($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $HBLog -Value ("[{0}] {1}" -f $ts, $msg)
}

function Get-BotProc {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match "execution\.orchestrator" }
}

# ── 1. Check log freshness ────────────────────────────────────────────────────
$logStale = $true
if (Test-Path $LogFile) {
    $age = (Get-Date) - (Get-Item $LogFile).LastWriteTime
    $logStale = $age.TotalMinutes -gt $StaleMinutes
}

# ── 2. Check process ──────────────────────────────────────────────────────────
$procs = @(Get-BotProc)
$procDead = $procs.Count -eq 0

# ── 3. Action ─────────────────────────────────────────────────────────────────
if ($procDead -and -not $logStale) {
    # Process just died but log is recent — clean restart
    Write-HB "DEAD — process gone, log fresh. Restarting."
    Start-Process -FilePath $PythonCmd -ArgumentList "-m", "execution.orchestrator" `
        -WorkingDirectory $BotDir -WindowStyle Hidden
}
elseif ($logStale -and -not $procDead) {
    # Zombie: process alive but log hasn't been written in $StaleMinutes min
    Write-HB "ZOMBIE — process running but log stale ${StaleMinutes}+ min. Killing and restarting."
    $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 3
    Start-Process -FilePath $PythonCmd -ArgumentList "-m", "execution.orchestrator" `
        -WorkingDirectory $BotDir -WindowStyle Hidden
}
elseif ($procDead -and $logStale) {
    Write-HB "DEAD+STALE — full restart."
    Start-Process -FilePath $PythonCmd -ArgumentList "-m", "execution.orchestrator" `
        -WorkingDirectory $BotDir -WindowStyle Hidden
}
else {
    Write-HB "OK — process alive, log fresh ($([int]$age.TotalMinutes) min old)."
}

# ── 4. Ensure watchdog is also running ───────────────────────────────────────
$wdProc = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.CommandLine -match "watchdog\.ps1" }
if (-not $wdProc) {
    Write-HB "WATCHDOG dead — restarting watchdog.ps1."
    Start-Process -FilePath "powershell.exe" `
        -ArgumentList "-ExecutionPolicy", "Bypass", "-File", "$BotDir\watchdog.ps1" `
        -WorkingDirectory $BotDir -WindowStyle Hidden
}
