@echo off
echo Adding Windows Firewall rule for EcoWaste API Server...
netsh advfirewall firewall add rule name="EcoWaste API Server" dir=in action=allow protocol=TCP localport=8000
echo.
echo Done! The firewall rule has been added.
echo Your app should now be able to connect to the server at http://192.168.0.113:8000
echo.
pause
