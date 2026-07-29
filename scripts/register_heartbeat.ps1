param([string]$BotDir = "C:\Users\Administrator\trading-bot")

$TaskName  = "AiDEN-Heartbeat"
$Script    = $BotDir + "\scripts\heartbeat.ps1"
$ArgStr    = "-ExecutionPolicy Bypass -NonInteractive -WindowStyle Hidden -File " + $Script + " -BotDir " + $BotDir

$action    = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $ArgStr -WorkingDirectory $BotDir
$trigger   = New-ScheduledTaskTrigger -RepetitionInterval (New-TimeSpan -Minutes 5) -Once -At (Get-Date)
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$settings  = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 2) -StartWhenAvailable -AllowStartIfOnBatteries

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal | Out-Null

Start-ScheduledTask -TaskName $TaskName
Write-Host "Done. Heartbeat registered and started."
