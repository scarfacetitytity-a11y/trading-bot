# AiDEN VPS Bootstrap — Windows VPS (ForexVPS / Contabo Windows)
# Run as Administrator in PowerShell.
# Installs Python, Git, clones repo, sets up .env, registers three-process stack as Scheduled Tasks.

$ErrorActionPreference = "Stop"
$BOT_DIR    = "C:\aiden-bot"
$REPO_URL   = "https://github.com/scarfacetitytity-a11y/trading-bot.git"
$PYTHON_URL = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
$VENV       = "$BOT_DIR\venv"
$PY         = "$VENV\Scripts\python.exe"
$LOG_DIR    = "$BOT_DIR\logs"

Write-Host "=== AiDEN VPS Bootstrap ===" -ForegroundColor Cyan

# ── 1. Python ────────────────────────────────────────────────────────────────
$pythonOk = $false
try { $v = & python --version 2>&1; if ($v -match "3\.") { $pythonOk = $true; Write-Host "Python: $v" } } catch {}

if (-not $pythonOk) {
    $installer = "$env:TEMP\python-installer.exe"
    Write-Host "Downloading Python 3.11.9..."
    Invoke-WebRequest -Uri $PYTHON_URL -OutFile $installer -UseBasicParsing
    Start-Process -FilePath $installer -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0" -Wait
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")
    Write-Host "Python installed: $(& python --version 2>&1)" -ForegroundColor Green
}

# ── 2. Git ───────────────────────────────────────────────────────────────────
$gitOk = $false
try { & git --version | Out-Null; $gitOk = $true } catch {}

if (-not $gitOk) {
    $gi = "$env:TEMP\git-installer.exe"
    $GIT_URL = "https://github.com/git-for-windows/git/releases/download/v2.45.2.windows.1/Git-2.45.2-64-bit.exe"
    Write-Host "Downloading Git..."
    Invoke-WebRequest -Uri $GIT_URL -OutFile $gi -UseBasicParsing
    Start-Process -FilePath $gi -ArgumentList "/VERYSILENT /NORESTART /COMPONENTS=icons,ext\reg\shellhere,assoc,assoc_sh" -Wait
    $env:PATH = $env:PATH + ";C:\Program Files\Git\cmd"
    Write-Host "Git installed." -ForegroundColor Green
}

# ── 3. Clone / pull repo ─────────────────────────────────────────────────────
if (Test-Path "$BOT_DIR\.git") {
    Write-Host "Pulling latest..."
    & git -C $BOT_DIR pull
} else {
    Write-Host "Cloning repo..."
    & git clone $REPO_URL $BOT_DIR
}

# ── 4. Virtualenv + dependencies ─────────────────────────────────────────────
if (-not (Test-Path $VENV)) { & python -m venv $VENV }
& "$VENV\Scripts\pip.exe" install --upgrade pip --quiet
& "$VENV\Scripts\pip.exe" install -r "$BOT_DIR\requirements.txt" --quiet
Write-Host "Dependencies installed." -ForegroundColor Green

# ── 5. Create .env if missing ────────────────────────────────────────────────
$envFile = "$BOT_DIR\.env"
if (-not (Test-Path $envFile)) {
    New-Item -ItemType File -Path $envFile | Out-Null
    @"
# MT5 Demo account (FTMO $100k challenge — account 1514131398)
MT5_LOGIN=YOUR_MT5_LOGIN
MT5_PASSWORD=YOUR_MT5_PASSWORD
MT5_SERVER=FTMO-Demo

# Telegram alerts (rajanmusicemail@gmail.com bot)
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN
TELEGRAM_CHAT_ID=YOUR_CHAT_ID
"@ | Out-File -FilePath $envFile -Encoding utf8
    Write-Host ".env created — fill in credentials before starting." -ForegroundColor Yellow
} else {
    Write-Host ".env already exists." -ForegroundColor Green
}

# ── 6. Logs directory ────────────────────────────────────────────────────────
New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null

# ── 7. Scheduled Tasks (auto-start on reboot) ────────────────────────────────
# Architecture: watchdog starts bot automatically. Register watchdog + code_monitor only.

$tasks = @(
    @{
        Name      = "AiDEN-Watchdog"
        Argument  = "-m execution.watchdog"
        StdOut    = "$LOG_DIR\watchdog.log"
        StdErr    = "$LOG_DIR\watchdog_err.log"
    },
    @{
        Name      = "AiDEN-CodeMonitor"
        Argument  = "-m execution.code_monitor"
        StdOut    = "$LOG_DIR\code_monitor.log"
        StdErr    = "$LOG_DIR\code_monitor_err.log"
    }
)

foreach ($t in $tasks) {
    $logStdOut = $t.StdOut
    $logStdErr = $t.StdErr
    # Wrap in cmd.exe so stdout/stderr redirect works in Scheduled Tasks
    $cmd  = "cmd.exe"
    $args = "/c `"$PY $($t.Argument) >> $logStdOut 2>> $logStdErr`""

    $action   = New-ScheduledTaskAction -Execute $cmd -Argument $args -WorkingDirectory $BOT_DIR
    $trigger  = New-ScheduledTaskTrigger -AtStartup
    $settings = New-ScheduledTaskSettingsSet -RestartCount 10 -RestartInterval (New-TimeSpan -Minutes 2) `
                    -ExecutionTimeLimit (New-TimeSpan -Days 0) -MultipleInstances IgnoreNew
    $principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

    if (Get-ScheduledTask -TaskName $t.Name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $t.Name -Confirm:$false
    }
    Register-ScheduledTask -TaskName $t.Name -Action $action -Trigger $trigger `
        -Settings $settings -Principal $principal | Out-Null
    Write-Host "Registered task: $($t.Name)" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Bootstrap complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Edit $envFile — fill in MT5 credentials + Telegram tokens"
Write-Host "  2. Edit $BOT_DIR\config\config_vps.yaml if MT5 terminal path differs"
Write-Host "     (default: C:\Program Files\MetaTrader 5\terminal64.exe)"
Write-Host "  3. Reboot VPS — both tasks start automatically, watchdog launches bot"
Write-Host "     OR start now:"
Write-Host "       Start-ScheduledTask -TaskName AiDEN-Watchdog"
Write-Host "       Start-ScheduledTask -TaskName AiDEN-CodeMonitor"
Write-Host ""
Write-Host "  Logs:"
Write-Host "    Bot:          $LOG_DIR\orchestrator.log"
Write-Host "    Watchdog:     $LOG_DIR\watchdog.log"
Write-Host "    Code monitor: $LOG_DIR\code_monitor.log"
