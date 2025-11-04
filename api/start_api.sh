#!/bin/bash
cd "$(dirname "$0")"
echo ""
echo "============================================================"
echo " Starting EcoWaste AI API Server"
echo "============================================================"
echo ""
echo "Server will be available at: http://192.168.0.113:8000"
echo "Press CTRL+C to stop the server"
echo ""
python real_api.py
