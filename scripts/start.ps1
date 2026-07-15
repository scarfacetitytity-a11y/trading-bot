# AiDEN Bot — Windows VPS daily startup
# Pulls latest code from GitHub, then starts the orchestrator
# Designed to run on server restart or manually

param(
    [string]$BotDir   = "C:\aiden-bot",
    [switch]$DryRun,
    [string]$Symbol   = ""
)

Set-Location $BotDir

Write-Host "=== AiDEN Bot Starting ===" -ForegroundColor Cyan
Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss UTC')"

# 1. Pull latest from GitHub
Write-Host "Pulling latest from GitHub..."
git pull origin master
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: git pull failed — running on existing code." -ForegroundColor Yellow
}

# 2. Install any new dependencies
Write-Host "Checking dependencies..."
& "$BotDir\venv\Scripts\pip.exe" install -r requirements.txt --quiet

# 3. Build orchestrator args
$args = @()
if ($DryRun)    { $args += "--dry-run" }
if ($Symbol)    { $args += "--symbol"; $args += $Symbol }

# 4. Start bot — restarts automatically on crash (loop)
$restarts = 0
while ($true) {
    Write-Host "Starting orchestrator (restart #$restarts)..." -ForegroundColor Green
    & "$BotDir\venv\Scripts\python.exe" -m execution.orchestrator @args
    $exit = $LASTEXITCODE
    Write-Host "Orchestrator exited with code $exit at $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Yellow

    # Intentional exit codes — don't restart
    if ($exit -eq 0) { Write-Host "Clean shutdown."; break }

    $restarts++
    if ($restarts -ge 10) {
        Write-Host "Too many restarts ($restarts). Stopping." -ForegroundColor Red
        break
    }

    $wait = [Math]::Min(30 * $restarts, 300)
    Write-Host "Restarting in ${wait}s..."
    Start-Sleep -Seconds $wait
}
