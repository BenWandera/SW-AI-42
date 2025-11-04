@echo off
echo 🔒 Turning Windows Firewall back ON...
echo.
netsh advfirewall set allprofiles state on
echo.
echo ✅ Firewall is now ON and your system is protected!
echo.
pause
