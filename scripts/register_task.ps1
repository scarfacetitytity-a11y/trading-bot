# Register AiDEN Bot as a Windows Scheduled Task
# Runs at system startup, hidden (no terminal window), as SYSTEM account
# Re-run this any time you need to update the task registration
# Usage: Run as Administrator
#   .\scripts\register_task.ps1
#   .\scripts\register_task.ps1 -BotDir C:\aiden-bot

param(
    [string]$BotDir  = "C:\aiden-bot",
    [string]$TaskName = "AiDEN-Bot"
)

$ErrorActionPreference = "Stop"

# Wrapper vbs — launches start.ps1 completely hidden (no console window at all)
$vbsPath = "$BotDir\scripts\start_hidden.vbs"
$vbsContent = @"
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "powershell.exe -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File $BotDir\scripts\start.ps1 -BotDir $BotDir", 0, False
"@
Set-Content -Path $vbsPath -Value $vbsContent -Encoding ASCII

# Scheduled task: triggers at boot + if it crashes, retry every 2 min up to 10x
$action   = New-ScheduledTaskAction `
    -Execute "wscript.exe" `
    -Argument "`"$vbsPath`""

$trigger  = New-ScheduledTaskTrigger -AtStartup

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0) `
    -RestartCount 10 `
    -RestartInterval (New-TimeSpan -Minutes 2) `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

# Run as SYSTEM — survives logoff, no password required
$principal = New-ScheduledTaskPrincipal `
    -UserId "SYSTEM" `
    -LogonType ServiceAccount `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName  $TaskName `
    -Action    $action `
    -Trigger   $trigger `
    -Settings  $settings `
    -Principal $principal `
    -Force | Out-Null

Write-Host "Task '$TaskName' registered." -ForegroundColor Green
Write-Host "  Start now:  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "  Stop:       Stop-ScheduledTask  -TaskName '$TaskName'"
Write-Host "  Status:     Get-ScheduledTask   -TaskName '$TaskName' | Select-Object State"
Write-Host "  Remove:     Unregister-ScheduledTask -TaskName '$TaskName'"
