' Silent launcher for Project Glitch Dashboard
' Runs the batch file hidden (no console window)

Set WshShell = CreateObject("WScript.Shell")
WshShell.Run """C:\Users\Naff\project-glitch\start-dashboard.bat""", 0, False
