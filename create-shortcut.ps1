$startupDir = [Environment]::GetFolderPath('Startup')
$shortcutPath = Join-Path $startupDir 'ProjectGlitch.lnk'
$targetPath = 'C:\Users\Naff\project-glitch\start-dashboard.vbs'

$WshShell = New-Object -ComObject WScript.Shell
$shortcut = $WshShell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $targetPath
$shortcut.WorkingDirectory = 'C:\Users\Naff\project-glitch'
$shortcut.Description = 'Project Glitch Dashboard'
$shortcut.WindowStyle = 7
$shortcut.Save()

Write-Host "Shortcut created at: $shortcutPath"
