#!/bin/bash

# Test script for CAISO streaming pipeline
# Tests Redis, streaming service, WebSocket server, and dashboard

echo "=========================================="
echo "CAISO Streaming Pipeline Test"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Redis is installed
echo -e "\n${YELLOW}1. Checking Redis installation...${NC}"
if command -v redis-cli &> /dev/null; then
    echo -e "${GREEN}✓ Redis is installed${NC}"
else
    echo -e "${RED}✗ Redis is not installed${NC}"
    echo "Install with: brew install redis (macOS) or apt-get install redis-server (Linux)"
    exit 1
fi

# Check if Redis is running
echo -e "\n${YELLOW}2. Checking Redis server...${NC}"
if redis-cli ping &> /dev/null; then
    echo -e "${GREEN}✓ Redis is running${NC}"
else
    echo -e "${RED}✗ Redis is not running${NC}"
    echo "Start with: redis-server"
    exit 1
fi

# Install Python dependencies
echo -e "\n${YELLOW}3. Installing Python dependencies...${NC}"
pip install -q -r requirements_streaming.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Set up Redis data structures
echo -e "\n${YELLOW}4. Setting up Redis data structures...${NC}"
python setup_redis.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Redis setup complete${NC}"
else
    echo -e "${RED}✗ Redis setup failed${NC}"
    exit 1
fi

# Start streaming service in background
echo -e "\n${YELLOW}5. Starting CAISO streaming service...${NC}"
python caiso_streaming_service.py &
STREAM_PID=$!
sleep 3

# Check if streaming service is running
if ps -p $STREAM_PID > /dev/null; then
    echo -e "${GREEN}✓ Streaming service started (PID: $STREAM_PID)${NC}"
else
    echo -e "${RED}✗ Streaming service failed to start${NC}"
    exit 1
fi

# Start WebSocket server in background
echo -e "\n${YELLOW}6. Starting WebSocket server...${NC}"
python websocket_server.py &
WS_PID=$!
sleep 3

# Check if WebSocket server is running
if ps -p $WS_PID > /dev/null; then
    echo -e "${GREEN}✓ WebSocket server started (PID: $WS_PID)${NC}"
else
    echo -e "${RED}✗ WebSocket server failed to start${NC}"
    kill $STREAM_PID
    exit 1
fi

# Test WebSocket endpoint
echo -e "\n${YELLOW}7. Testing WebSocket API...${NC}"
curl -s http://localhost:8000/api/current > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ WebSocket API is responding${NC}"
else
    echo -e "${RED}✗ WebSocket API is not responding${NC}"
fi

# Display dashboard URL
echo -e "\n${YELLOW}8. Dashboard ready!${NC}"
echo -e "${GREEN}=========================================="
echo -e "✓ All services are running!"
echo -e "==========================================${NC}"
echo ""
echo "Dashboard URL: http://localhost:8000"
echo "Or open: file://$(pwd)/dashboard.html"
echo ""
echo "Service PIDs:"
echo "  Streaming: $STREAM_PID"
echo "  WebSocket: $WS_PID"
echo ""
echo "To stop all services, run:"
echo "  kill $STREAM_PID $WS_PID"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

# Wait for interrupt
trap "echo -e '\n${YELLOW}Stopping services...${NC}'; kill $STREAM_PID $WS_PID 2>/dev/null; echo -e '${GREEN}Services stopped${NC}'; exit 0" INT

while true; do
    sleep 1
done