$action = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument 'C:\Users\Naff\project-glitch\start-dashboard.vbs'
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -RunLevel Highest
Register-ScheduledTask -TaskName 'ProjectGlitch' -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force
Write-Host "Scheduled task created successfully"
