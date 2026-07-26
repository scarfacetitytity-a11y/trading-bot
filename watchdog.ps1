# AiDEN bot watchdog — restarts orchestrator if it dies.
# Run this in a separate PowerShell window, or schedule it via Task Scheduler.
# Usage: powershell -ExecutionPolicy Bypass -File watchdog.ps1

$BotDir   = $PSScriptRoot
$PythonCmd = "python"
$Module   = "execution.orchestrator"
$LogFile  = "$BotDir\logs\watchdog.log"
$MaxRestarts = 20
$CooldownSec = 30

$restarts = 0

function Write-Log($msg) {
    $ts = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

Write-Log "Watchdog started. Monitoring AiDEN orchestrator."

while ($restarts -lt $MaxRestarts) {
    Write-Log "Starting orchestrator (attempt $($restarts + 1)/$MaxRestarts)..."

    $proc = Start-Process -FilePath $PythonCmd `
        -ArgumentList "-m", $Module `
        -WorkingDirectory $BotDir `
        -PassThru `
        -NoNewWindow

    Write-Log "Orchestrator PID $($proc.Id) started."
    $proc.WaitForExit()

    $exitCode = $proc.ExitCode
    Write-Log "Orchestrator exited with code $exitCode."

    if ($exitCode -eq 0) {
        Write-Log "Clean exit (code 0) — watchdog stopping."
        break
    }

    $restarts++
    if ($restarts -ge $MaxRestarts) {
        Write-Log "Max restarts ($MaxRestarts) reached — watchdog giving up. Check logs."
        break
    }

    Write-Log "Waiting ${CooldownSec}s before restart..."
    Start-Sleep -Seconds $CooldownSec
}
