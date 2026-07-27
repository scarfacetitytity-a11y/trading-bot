# AiDEN watchdog - supervises orchestrator AND council_watch.
# Checks every 30s; starts whichever is not running; kills duplicate bots.
# Safe at boot: checks the process table before launching, and the
# orchestrator itself aborts on a live PID lock.
# Usage: powershell -ExecutionPolicy Bypass -File watchdog.ps1

$BotDir    = $PSScriptRoot
$PythonCmd = "C:\Python314\python.exe"
$LogFile   = Join-Path $BotDir "logs\watchdog.log"
$PollSec   = 30

function Write-Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogFile -Value ("[{0}] {1}" -f $ts, $msg)
}

function Get-BotProcess($module) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match [regex]::Escape($module) }
}

Write-Log "Watchdog started - supervising orchestrator + council_watch."

while ($true) {
    foreach ($module in @("execution.orchestrator", "execution.council_watch")) {
        $procs = @(Get-BotProcess $module)
        if ($procs.Count -eq 0) {
            Write-Log ("{0} not running - launching." -f $module)
            Start-Process -FilePath $PythonCmd -ArgumentList "-m", $module -WorkingDirectory $BotDir -WindowStyle Hidden
        }
        elseif ($module -eq "execution.orchestrator" -and $procs.Count -gt 1) {
            Write-Log ("WARNING: {0} orchestrator instances - killing extras." -f $procs.Count)
            $procs | Sort-Object CreationDate | Select-Object -Skip 1 | ForEach-Object {
                Write-Log ("Killing duplicate orchestrator PID {0}" -f $_.ProcessId)
                Stop-Process -Id $_.ProcessId -Force
            }
        }
    }
    Start-Sleep -Seconds $PollSec
}
