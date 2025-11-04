@echo off
echo ⚠️  TEMPORARY: Turning off Windows Firewall for testing...
echo.
netsh advfirewall set allprofiles state off
echo.
echo ✅ Firewall is now OFF. Test your app now!
echo.
echo ⚠️  IMPORTANT: Run 'temp_enable_firewall.bat' when done to turn it back on!
echo.
pause
