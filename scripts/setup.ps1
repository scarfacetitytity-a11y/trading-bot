# AiDEN Bot — Windows VPS first-time setup
# Run once as Administrator on a fresh Windows Server
# Usage: .\scripts\setup.ps1

param(
    [string]$RepoUrl  = "https://github.com/scarfacetitytity-a11y/trading-bot.git",
    [string]$InstallDir = "C:\aiden-bot"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "=== AiDEN Bot Setup ===" -ForegroundColor Cyan

# 1. Clone repo
if (-not (Test-Path $InstallDir)) {
    Write-Host "Cloning repo to $InstallDir..."
    git clone $RepoUrl $InstallDir
} else {
    Write-Host "Repo already exists at $InstallDir — pulling latest..."
    Set-Location $InstallDir
    git pull origin master
}

Set-Location $InstallDir

# 2. Python venv
if (-not (Test-Path "$InstallDir\venv")) {
    Write-Host "Creating Python venv..."
    python -m venv venv
}

# 3. Install dependencies
Write-Host "Installing dependencies..."
& "$InstallDir\venv\Scripts\pip.exe" install -r requirements.txt --quiet

# 4. Config file
if (-not (Test-Path "$InstallDir\config\config.yaml")) {
    Write-Host "Creating config from example..."
    Copy-Item "$InstallDir\config.example.yaml" "$InstallDir\config\config.yaml"
    Write-Host "EDIT config\config.yaml before starting the bot." -ForegroundColor Yellow
}

# 5. Logs dir
New-Item -ItemType Directory -Force -Path "$InstallDir\logs" | Out-Null

# 6. Register as Windows scheduled task (hidden, no terminal window)
& "$InstallDir\scripts\register_task.ps1" -BotDir $InstallDir

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "  1. Edit config\config.yaml (MT5 terminal path)"
Write-Host "  2. Open MT5 and log into your FTMO account"
Write-Host "  3. Enable Algo Trading in MT5 toolbar"
Write-Host "  4. Run: .\scripts\start.ps1"
