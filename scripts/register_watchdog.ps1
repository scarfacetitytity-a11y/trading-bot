# Registers the AiDEN watchdog as a Task Scheduler job: starts at logon,
# restarts if it dies. Run once:
#   powershell -ExecutionPolicy Bypass -File scripts\register_watchdog.ps1

$BotDir   = Split-Path $PSScriptRoot -Parent
$TaskName = "AiDEN-Watchdog"

$argStr   = '-WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}\watchdog.ps1"' -f $BotDir
$action   = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argStr -WorkingDirectory $BotDir
$trigger  = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Days 3650) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings | Out-Null
Write-Host ("Registered {0} - starts at logon, auto-restarts. Start now: Start-ScheduledTask -TaskName {0}" -f $TaskName)
