# Registers AiDEN heartbeat as a Task Scheduler job — runs every 5 minutes, always.
# Survives reboots, logoffs, and VPS restarts without needing a user session.
# Run once on the VPS (as Administrator):
#   powershell -ExecutionPolicy Bypass -File scripts\register_heartbeat.ps1

param([string]$BotDir = "C:\aiden-bot")

$TaskName = "AiDEN-Heartbeat"
$Script   = "$BotDir\scripts\heartbeat.ps1"
$ArgStr   = "-ExecutionPolicy Bypass -NonInteractive -WindowStyle Hidden -File `"$Script`" -BotDir `"$BotDir`""

$action   = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $ArgStr -WorkingDirectory $BotDir

# Trigger: every 5 minutes, starting now, repeat forever
$trigger  = New-ScheduledTaskTrigger -RepetitionInterval (New-TimeSpan -Minutes 5) -Once -At (Get-Date)

# Run whether or not user is logged in; highest privileges
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 2) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal | Out-Null

Write-Host "Registered '$TaskName' — runs every 5 min as SYSTEM, survives reboots."
Write-Host "Force a run now: Start-ScheduledTask -TaskName '$TaskName'"
