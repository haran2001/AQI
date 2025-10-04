#!/usr/bin/env python3
"""
WebSocket Server for Real-Time CAISO Dashboard

Subscribes to Redis streams and broadcasts price updates to connected dashboard clients.
Supports multiple concurrent connections with automatic reconnection.

Architecture:
- FastAPI for HTTP endpoints
- WebSocket for real-time updates
- Redis Pub/Sub for data source
- JSON for message format

Usage:
    python websocket_server.py
"""

import json
import redis
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        """Send message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            if connection in self.active_connections:
                self.active_connections.remove(connection)


class RedisSubscriber:
    """Subscribes to Redis channels and forwards to WebSocket clients"""

    def __init__(self, manager: ConnectionManager):
        self.manager = manager
        self.redis_client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True
        )
        self.pubsub = self.redis_client.pubsub()
        self.running = False

    async def subscribe_and_forward(self):
        """Subscribe to Redis channels and forward messages to WebSocket clients"""
        try:
            # Subscribe to channels
            self.pubsub.subscribe('caiso:sp15:updates', 'caiso:alerts')
            self.running = True

            logger.info("Started Redis subscription")

            while self.running:
                # Check for messages (non-blocking)
                message = self.pubsub.get_message(timeout=1.0)

                if message and message['type'] == 'message':
                    channel = message['channel']
                    data = message['data']

                    # Parse and enhance message
                    try:
                        price_data = json.loads(data)
                        price_data['channel'] = channel
                        price_data['server_time'] = datetime.utcnow().isoformat()

                        # Broadcast to all WebSocket clients
                        await self.manager.broadcast(json.dumps(price_data))

                        logger.debug(f"Broadcasted {channel} update to {len(self.manager.active_connections)} clients")

                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON from Redis: {data}")

                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Redis subscriber error: {e}")
        finally:
            self.pubsub.close()
            self.running = False

    def stop(self):
        """Stop the subscription"""
        self.running = False

    def get_current_prices(self) -> Dict[str, Any]:
        """Get current prices from Redis"""
        try:
            current = self.redis_client.hgetall('caiso:sp15:current')
            return current
        except Exception as e:
            logger.error(f"Error getting current prices: {e}")
            return {}

    def get_historical_prices(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical prices from Redis"""
        try:
            # Calculate time range
            end_time = datetime.utcnow().timestamp()
            start_time = (datetime.utcnow() - timedelta(hours=hours)).timestamp()

            # Get RT prices from sorted set
            rt_prices = self.redis_client.zrangebyscore(
                'caiso:sp15:rt:5min',
                start_time,
                end_time,
                withscores=True
            )

            # Parse results
            historical = []
            for item, score in rt_prices:
                parts = item.split(':')
                if len(parts) == 2:
                    timestamp_val, price = parts
                    historical.append({
                        'timestamp': datetime.fromtimestamp(float(timestamp_val)).isoformat(),
                        'price': float(price),
                        'type': 'RT'
                    })

            return historical

        except Exception as e:
            logger.error(f"Error getting historical prices: {e}")
            return []


# Create managers
manager = ConnectionManager()
redis_subscriber = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global redis_subscriber

    # Startup
    redis_subscriber = RedisSubscriber(manager)
    asyncio.create_task(redis_subscriber.subscribe_and_forward())
    logger.info("WebSocket server started")

    yield

    # Shutdown
    if redis_subscriber:
        redis_subscriber.stop()
    logger.info("WebSocket server stopped")


# Create FastAPI app
app = FastAPI(title="CAISO Real-Time Dashboard", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get():
    """Serve the dashboard HTML"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CAISO Real-Time Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            #status { padding: 10px; margin: 10px 0; }
            .connected { background: #d4edda; color: #155724; }
            .disconnected { background: #f8d7da; color: #721c24; }
            #prices { margin: 20px 0; }
            .price-box { display: inline-block; padding: 20px; margin: 10px; border: 1px solid #ddd; }
            #messages { height: 300px; overflow-y: scroll; border: 1px solid #ddd; padding: 10px; }
            .message { margin: 5px 0; padding: 5px; background: #f0f0f0; }
        </style>
    </head>
    <body>
        <h1>CAISO SP15 Real-Time Prices</h1>

        <div id="status" class="disconnected">Disconnected</div>

        <div id="prices">
            <div class="price-box">
                <h3>Real-Time Price</h3>
                <div id="rt-price">--</div>
                <small id="rt-time">--</small>
            </div>
            <div class="price-box">
                <h3>Day-Ahead Price</h3>
                <div id="da-price">--</div>
                <small id="da-time">--</small>
            </div>
        </div>

        <h3>Live Updates</h3>
        <div id="messages"></div>

        <script>
            const ws = new WebSocket("ws://localhost:8000/ws");
            const status = document.getElementById('status');
            const messages = document.getElementById('messages');
            const rtPrice = document.getElementById('rt-price');
            const rtTime = document.getElementById('rt-time');
            const daPrice = document.getElementById('da-price');
            const daTime = document.getElementById('da-time');

            ws.onopen = function(event) {
                status.className = 'connected';
                status.textContent = 'Connected';
                addMessage('Connected to server');
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);

                if (data.type === 'RT') {
                    rtPrice.textContent = '$' + data.price.toFixed(2) + '/MWh';
                    rtTime.textContent = new Date(data.timestamp).toLocaleString();
                } else if (data.type === 'DA') {
                    daPrice.textContent = '$' + data.price.toFixed(2) + '/MWh';
                    daTime.textContent = new Date(data.timestamp).toLocaleString();
                }

                addMessage(`${data.type} Price: $${data.price.toFixed(2)}/MWh at ${new Date(data.timestamp).toLocaleTimeString()}`);
            };

            ws.onclose = function(event) {
                status.className = 'disconnected';
                status.textContent = 'Disconnected';
                addMessage('Disconnected from server');

                // Reconnect after 5 seconds
                setTimeout(() => {
                    window.location.reload();
                }, 5000);
            };

            function addMessage(text) {
                const msg = document.createElement('div');
                msg.className = 'message';
                msg.textContent = new Date().toLocaleTimeString() + ': ' + text;
                messages.insertBefore(msg, messages.firstChild);

                // Keep only last 50 messages
                while (messages.children.length > 50) {
                    messages.removeChild(messages.lastChild);
                }
            }
        </script>
    </body>
    </html>
    """)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)

    try:
        # Send current prices on connect
        if redis_subscriber:
            current = redis_subscriber.get_current_prices()
            if current:
                await websocket.send_text(json.dumps({
                    'type': 'current',
                    'data': current,
                    'timestamp': datetime.utcnow().isoformat()
                }))

            # Send recent history
            history = redis_subscriber.get_historical_prices(hours=1)
            if history:
                await websocket.send_text(json.dumps({
                    'type': 'history',
                    'data': history,
                    'timestamp': datetime.utcnow().isoformat()
                }))

        # Keep connection alive and wait for messages
        while True:
            # Wait for any message from client (ping/pong)
            data = await websocket.receive_text()

            # Echo back as pong
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/current")
async def get_current_prices():
    """REST API endpoint for current prices"""
    if redis_subscriber:
        return redis_subscriber.get_current_prices()
    return {"error": "Redis not connected"}


@app.get("/api/history/{hours}")
async def get_historical_prices(hours: int = 24):
    """REST API endpoint for historical prices"""
    if redis_subscriber:
        return {
            "hours": hours,
            "data": redis_subscriber.get_historical_prices(hours)
        }
    return {"error": "Redis not connected"}


if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )