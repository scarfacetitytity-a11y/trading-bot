# AiDEN heartbeat — runs every 5 minutes via Task Scheduler.
# Checks process alive + log freshness. Sends Telegram alert on any issue or restart.
# Register with: powershell -ExecutionPolicy Bypass -File scripts\register_heartbeat.ps1

param(
    [string]$BotDir       = "C:\aiden-bot",
    [int]   $StaleMinutes = 10
)

$PythonCmd = "$BotDir\venv\Scripts\python.exe"
$LogFile   = "$BotDir\logs\bot.log"
$HBLog     = "$BotDir\logs\heartbeat.log"
$EnvFile   = "$BotDir\.env"

# ── Load .env ─────────────────────────────────────────────────────────────────
$TgToken  = ""
$TgChatId = ""
if (Test-Path $EnvFile) {
    foreach ($line in Get-Content $EnvFile) {
        if ($line -match "^TELEGRAM_BOT_TOKEN\s*=\s*(.+)$")  { $TgToken  = $Matches[1].Trim() }
        if ($line -match "^TELEGRAM_CHAT_ID\s*=\s*(.+)$")    { $TgChatId = $Matches[1].Trim() }
    }
}

# ── Helpers ───────────────────────────────────────────────────────────────────
function Write-HB($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $HBLog -Value ("[{0}] {1}" -f $ts, $msg)
}

function Send-Tg($text) {
    if (-not $TgToken -or -not $TgChatId) { return }
    try {
        $body = @{
            chat_id    = $TgChatId
            text       = $text
            parse_mode = "HTML"
        } | ConvertTo-Json
        $url = "https://api.telegram.org/bot$TgToken/sendMessage"
        Invoke-RestMethod -Uri $url -Method Post -Body $body -ContentType "application/json" -TimeoutSec 8 | Out-Null
    } catch {
        Write-HB "Telegram send failed: $_"
    }
}

function Get-BotProc {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match "execution\.orchestrator" }
}

function Start-Bot {
    Start-Process -FilePath $PythonCmd -ArgumentList "-m", "execution.orchestrator" `
        -WorkingDirectory $BotDir -WindowStyle Hidden
}

# ── Checks ────────────────────────────────────────────────────────────────────
$procs    = @(Get-BotProc)
$procDead = $procs.Count -eq 0

$logStale = $true
$ageMin   = 999
if (Test-Path $LogFile) {
    $age      = (Get-Date) - (Get-Item $LogFile).LastWriteTime
    $ageMin   = [int]$age.TotalMinutes
    $logStale = $ageMin -gt $StaleMinutes
}

$ts = Get-Date -Format "HH:mm UTC"

# ── Action + Telegram alert ───────────────────────────────────────────────────
if ($procDead -and $logStale) {
    $msg = "DEAD+STALE — process gone, log $ageMin min old. Full restart."
    Write-HB $msg
    Send-Tg "🔴 <b>AiDEN BOT DEAD</b>`n`nProcess: gone | Log: ${ageMin}min old`nRestarting now...`n<i>$ts</i>"
    Start-Bot

} elseif ($procDead) {
    $msg = "DEAD — process gone (log fresh $ageMin min). Restarting."
    Write-HB $msg
    Send-Tg "🔴 <b>AiDEN BOT CRASHED</b>`n`nProcess died — log was ${ageMin}min old.`nRestarting now...`n<i>$ts</i>"
    Start-Bot

} elseif ($logStale) {
    $msg = "ZOMBIE — process running but log stale ${ageMin}min. Killing + restarting."
    Write-HB $msg
    Send-Tg "⚠️ <b>AiDEN BOT ZOMBIE</b>`n`nProcess alive but log not updated for ${ageMin}min.`nKilling and restarting...`n<i>$ts</i>"
    $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 3
    Start-Bot

} else {
    Write-HB "OK — alive, log ${ageMin}min old."
    # No Telegram on every OK — too noisy. Alert only on events above.
}

# ── Ensure watchdog is also alive ─────────────────────────────────────────────
$wdProc = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.CommandLine -match "watchdog\.ps1" }
if (-not $wdProc) {
    Write-HB "WATCHDOG dead — restarting."
    Send-Tg "⚠️ <b>AiDEN Watchdog Restarted</b>`n`nwatchdog.ps1 was not running — restarted.`n<i>$ts</i>"
    Start-Process -FilePath "powershell.exe" `
        -ArgumentList "-ExecutionPolicy", "Bypass", "-File", "$BotDir\watchdog.ps1" `
        -WorkingDirectory $BotDir -WindowStyle Hidden
}
