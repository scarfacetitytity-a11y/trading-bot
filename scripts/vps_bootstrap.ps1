# AiDEN VPS Bootstrap — ForexVPS Windows
# Paste this entire script into the VPS PowerShell terminal and run it.
# Run as Administrator.

$ErrorActionPreference = "Stop"
$BOT_DIR = "C:\aiden-bot"
$REPO_URL = "https://github.com/scarfacetitytity-a11y/trading-bot.git"
$PYTHON_URL = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
$PYTHON_INSTALLER = "$env:TEMP\python-installer.exe"
$TASK_NAME = "AiDEN-Bot"

Write-Host "=== AiDEN VPS Bootstrap ===" -ForegroundColor Cyan

# ── 1. Python ────────────────────────────────────────────────────────────────
$pythonOk = $false
try {
    $v = & python --version 2>&1
    if ($v -match "3\.") { $pythonOk = $true; Write-Host "Python already installed: $v" }
} catch {}

if (-not $pythonOk) {
    Write-Host "Downloading Python 3.11.9..."
    Invoke-WebRequest -Uri $PYTHON_URL -OutFile $PYTHON_INSTALLER -UseBasicParsing
    Write-Host "Installing Python (adds to PATH)..."
    Start-Process -FilePath $PYTHON_INSTALLER -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0" -Wait
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
    $v = & python --version 2>&1
    Write-Host "Installed: $v" -ForegroundColor Green
}

# ── 2. Git ───────────────────────────────────────────────────────────────────
$gitOk = $false
try { & git --version | Out-Null; $gitOk = $true } catch {}

if (-not $gitOk) {
    $GIT_URL = "https://github.com/git-for-windows/git/releases/download/v2.45.2.windows.1/Git-2.45.2-64-bit.exe"
    $GIT_INSTALLER = "$env:TEMP\git-installer.exe"
    Write-Host "Downloading Git..."
    Invoke-WebRequest -Uri $GIT_URL -OutFile $GIT_INSTALLER -UseBasicParsing
    Start-Process -FilePath $GIT_INSTALLER -ArgumentList "/VERYSILENT /NORESTART /COMPONENTS=icons,ext\reg\shellhere,assoc,assoc_sh" -Wait
    $env:PATH = $env:PATH + ";C:\Program Files\Git\cmd"
    Write-Host "Git installed." -ForegroundColor Green
}

# ── 3. Clone repo ────────────────────────────────────────────────────────────
if (Test-Path $BOT_DIR) {
    Write-Host "Bot directory exists — pulling latest..."
    & git -C $BOT_DIR pull
} else {
    Write-Host "Cloning repo..."
    & git clone $REPO_URL $BOT_DIR
}

# ── 4. Virtual environment + dependencies ────────────────────────────────────
Write-Host "Setting up Python environment..."
if (-not (Test-Path "$BOT_DIR\venv")) {
    & python -m venv "$BOT_DIR\venv"
}
& "$BOT_DIR\venv\Scripts\pip.exe" install --upgrade pip --quiet
& "$BOT_DIR\venv\Scripts\pip.exe" install -r "$BOT_DIR\requirements.txt" --quiet
Write-Host "Dependencies installed." -ForegroundColor Green

# ── 5. Create .env if missing ────────────────────────────────────────────────
$envFile = "$BOT_DIR\.env"
if (-not (Test-Path $envFile)) {
    @"
# MT5 Demo account (100k FTMO challenge)
MT5_LOGIN_DEMO=YOUR_DEMO_LOGIN
MT5_PASSWORD_DEMO=YOUR_DEMO_PASSWORD
MT5_SERVER_DEMO=YOUR_BROKER_SERVER

# MT5 Live account (£50 live — separate instance)
MT5_LOGIN_LIVE=YOUR_LIVE_LOGIN
MT5_PASSWORD_LIVE=YOUR_LIVE_PASSWORD
MT5_SERVER_LIVE=YOUR_BROKER_SERVER_LIVE

# Telegram alerts
TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID=YOUR_TELEGRAM_CHAT_ID
"@ | Out-File -FilePath $envFile -Encoding utf8
    Write-Host ".env created — FILL IN YOUR CREDENTIALS before starting the bot." -ForegroundColor Yellow
} else {
    Write-Host ".env already exists — skipping." -ForegroundColor Green
}

# ── 6. Scheduled Task (auto-start on reboot) ─────────────────────────────────
Write-Host "Registering Windows Scheduled Task: $TASK_NAME..."
$action = New-ScheduledTaskAction `
    -Execute "$BOT_DIR\venv\Scripts\python.exe" `
    -Argument "$BOT_DIR\execution\orchestrator.py" `
    -WorkingDirectory $BOT_DIR

$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 2) -ExecutionTimeLimit (New-TimeSpan -Days 0)
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

if (Get-ScheduledTask -TaskName $TASK_NAME -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TASK_NAME -Confirm:$false
}
Register-ScheduledTask -TaskName $TASK_NAME -Action $action -Trigger $trigger -Settings $settings -Principal $principal | Out-Null
Write-Host "Scheduled Task registered — bot will auto-start on reboot." -ForegroundColor Green

Write-Host ""
Write-Host "=== Bootstrap complete ===" -ForegroundColor Cyan
Write-Host "Next steps:"
Write-Host "  1. Edit $envFile — fill in your MT5 credentials"
Write-Host "  2. Copy config\config.yaml to $BOT_DIR\config\config.yaml and verify terminal_path"
Write-Host "  3. Run: & '$BOT_DIR\venv\Scripts\python.exe' '$BOT_DIR\execution\orchestrator.py'"
Write-Host "  4. Or reboot the VPS — the Scheduled Task will start it automatically"
