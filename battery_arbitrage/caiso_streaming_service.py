#!/usr/bin/env python3
"""
CAISO Real-Time Streaming Service

Continuously fetches CAISO price data and streams to Redis for real-time dashboard.
Handles both Real-Time (5-min) and Day-Ahead (hourly) prices.

Architecture:
- Polls CAISO API at appropriate intervals
- Stores in Redis Streams for persistence
- Publishes to Redis Pub/Sub for real-time updates
- Maintains current state in Redis Hash

Usage:
    python caiso_streaming_service.py
"""

import os
import io
import sys
import json
import time
import redis
import signal
import logging
import zipfile
import requests
import traceback
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import CAISO helper functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from caiso_sp15_data_fetch import (
    _format_dt_for_oasis,
    _make_oasis_request,
    _parse_oasis_xml,
    SITE_INFO,
    OASIS_BASE
)


class CAISOStreamingService:
    """Streams CAISO price data to Redis in real-time"""

    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        """Initialize the streaming service"""
        # Redis connection
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )

        # Redis keys
        self.STREAM_KEY = 'caiso:sp15:stream'
        self.CURRENT_KEY = 'caiso:sp15:current'
        self.RT_TS_KEY = 'caiso:sp15:rt:5min'
        self.DA_TS_KEY = 'caiso:sp15:da:hourly'
        self.PUBSUB_CHANNEL = 'caiso:sp15:updates'
        self.ALERT_CHANNEL = 'caiso:alerts'

        # Polling intervals (seconds)
        self.RT_INTERVAL = 300  # 5 minutes
        self.DA_INTERVAL = 3600  # 1 hour

        # Last fetch timestamps
        self.last_rt_fetch = 0
        self.last_da_fetch = 0

        # Alert thresholds
        self.HIGH_PRICE_THRESHOLD = 100  # $/MWh
        self.NEGATIVE_PRICE_THRESHOLD = 0

        # Service control
        self.running = False

        logger.info(f"Initialized CAISO Streaming Service for {SITE_INFO['node']}")

    def fetch_latest_rt_price(self) -> Optional[Dict[str, Any]]:
        """Fetch latest Real-Time price from CAISO"""
        try:
            # Get current time and 1 hour ago
            now = datetime.utcnow()
            one_hour_ago = now - timedelta(hours=1)

            params = {
                "queryname": "PRC_INTVL_LMP",
                "market_run_id": "RTM",
                "node": SITE_INFO['node'],
                "startdatetime": _format_dt_for_oasis(one_hour_ago),
                "enddatetime": _format_dt_for_oasis(now),
                "version": "1"
            }

            logger.debug(f"Fetching RT prices from {one_hour_ago} to {now}")
            response = _make_oasis_request(OASIS_BASE, params)

            # Process response
            content = response.content
            if content[:2] == b'PK':  # ZIP file
                z = zipfile.ZipFile(io.BytesIO(content))
                files = z.namelist()
                if files:
                    data = z.read(files[0])
                else:
                    logger.warning("Empty ZIP file from CAISO")
                    return None
            else:
                data = content

            df = _parse_oasis_xml(data)

            if not df.empty and 'timestamp' in df.columns and 'price_mwh' in df.columns:
                # Get most recent price
                df = df.set_index('timestamp')
                df = df.sort_index()
                latest = df.iloc[-1]

                return {
                    'timestamp': df.index[-1].isoformat(),
                    'price': float(latest['price_mwh']),
                    'type': 'RT',
                    'node': SITE_INFO['node'],
                    'zone': SITE_INFO['zone']
                }

            logger.warning("No valid RT price data in response")
            return None

        except Exception as e:
            logger.error(f"Error fetching RT price: {e}")
            return None

    def fetch_latest_da_prices(self) -> Optional[Dict[str, Any]]:
        """Fetch latest Day-Ahead prices from CAISO"""
        try:
            # Get tomorrow's date for DA market
            tomorrow = datetime.utcnow().date() + timedelta(days=1)
            start_dt = datetime.combine(tomorrow, datetime.min.time())
            end_dt = start_dt + timedelta(days=1)

            params = {
                "queryname": "PRC_LMP",
                "market_run_id": "DAM",
                "node": SITE_INFO['node'],
                "startdatetime": _format_dt_for_oasis(start_dt),
                "enddatetime": _format_dt_for_oasis(end_dt),
                "version": "1"
            }

            logger.debug(f"Fetching DA prices for {tomorrow}")
            response = _make_oasis_request(OASIS_BASE, params)

            # Process response
            content = response.content
            if content[:2] == b'PK':  # ZIP file
                z = zipfile.ZipFile(io.BytesIO(content))
                files = z.namelist()
                if files:
                    data = z.read(files[0])
                else:
                    logger.warning("Empty ZIP file from CAISO")
                    return None
            else:
                data = content

            df = _parse_oasis_xml(data)

            if not df.empty and 'timestamp' in df.columns and 'price_mwh' in df.columns:
                # Return all DA prices for the day
                df = df.set_index('timestamp')
                df = df.sort_index()

                prices = []
                for timestamp, row in df.iterrows():
                    prices.append({
                        'timestamp': timestamp.isoformat(),
                        'price': float(row['price_mwh']),
                        'type': 'DA',
                        'node': SITE_INFO['node'],
                        'zone': SITE_INFO['zone']
                    })

                return {'prices': prices, 'date': tomorrow.isoformat()}

            logger.warning("No valid DA price data in response")
            return None

        except Exception as e:
            logger.error(f"Error fetching DA prices: {e}")
            return None

    def store_price_update(self, price_data: Dict[str, Any]):
        """Store price update in Redis"""
        try:
            timestamp_str = price_data['timestamp']
            timestamp = datetime.fromisoformat(timestamp_str).timestamp()
            price = price_data['price']
            price_type = price_data['type']

            # Add to Redis Stream
            stream_data = {
                'timestamp': timestamp_str,
                'price': price,
                'type': price_type,
                'node': price_data['node'],
                'zone': price_data['zone']
            }
            self.redis_client.xadd(self.STREAM_KEY, stream_data, maxlen=10000)

            # Update current price hash
            if price_type == 'RT':
                self.redis_client.hset(self.CURRENT_KEY, mapping={
                    'rt_price': price,
                    'rt_timestamp': timestamp_str,
                    'last_update': datetime.utcnow().isoformat()
                })

                # Add to time series sorted set
                self.redis_client.zadd(self.RT_TS_KEY, {f"{timestamp}:{price}": timestamp})

                # Trim to keep only last 7 days
                week_ago = datetime.utcnow() - timedelta(days=7)
                self.redis_client.zremrangebyscore(self.RT_TS_KEY, 0, week_ago.timestamp())

            elif price_type == 'DA':
                self.redis_client.hset(self.CURRENT_KEY, mapping={
                    'da_price': price,
                    'da_timestamp': timestamp_str
                })

                # Add to DA time series
                self.redis_client.zadd(self.DA_TS_KEY, {f"{timestamp}:{price}": timestamp})

                # Trim to keep only last 30 days
                month_ago = datetime.utcnow() - timedelta(days=30)
                self.redis_client.zremrangebyscore(self.DA_TS_KEY, 0, month_ago.timestamp())

            # Publish update to subscribers
            self.redis_client.publish(self.PUBSUB_CHANNEL, json.dumps(price_data))

            # Check for alerts
            self.check_alerts(price_data)

            logger.info(f"Stored {price_type} price: ${price:.2f}/MWh at {timestamp_str}")

        except Exception as e:
            logger.error(f"Error storing price update: {e}")

    def check_alerts(self, price_data: Dict[str, Any]):
        """Check for price alerts and publish if triggered"""
        price = price_data['price']

        alerts = []

        # High price alert
        if price > self.HIGH_PRICE_THRESHOLD:
            alerts.append({
                'type': 'high_price',
                'threshold': self.HIGH_PRICE_THRESHOLD,
                'price': price,
                'timestamp': price_data['timestamp']
            })

        # Negative price alert
        if price < self.NEGATIVE_PRICE_THRESHOLD:
            alerts.append({
                'type': 'negative_price',
                'price': price,
                'timestamp': price_data['timestamp']
            })

        # Publish alerts
        for alert in alerts:
            self.redis_client.publish(self.ALERT_CHANNEL, json.dumps(alert))
            logger.warning(f"Alert: {alert['type']} - Price: ${price:.2f}/MWh")

    def run_once(self):
        """Run one iteration of the streaming loop"""
        current_time = time.time()

        # Fetch RT prices every 5 minutes
        if current_time - self.last_rt_fetch >= self.RT_INTERVAL:
            logger.info("Fetching Real-Time prices...")
            rt_price = self.fetch_latest_rt_price()
            if rt_price:
                self.store_price_update(rt_price)
            self.last_rt_fetch = current_time

        # Fetch DA prices every hour
        if current_time - self.last_da_fetch >= self.DA_INTERVAL:
            logger.info("Fetching Day-Ahead prices...")
            da_result = self.fetch_latest_da_prices()
            if da_result:
                for price_data in da_result['prices']:
                    self.store_price_update(price_data)
            self.last_da_fetch = current_time

    def start(self):
        """Start the streaming service"""
        logger.info("Starting CAISO Streaming Service...")
        self.running = True

        # Initial fetch
        self.run_once()

        while self.running:
            try:
                # Calculate time until next update
                current_time = time.time()
                time_to_rt = self.RT_INTERVAL - (current_time - self.last_rt_fetch)
                time_to_da = self.DA_INTERVAL - (current_time - self.last_da_fetch)

                # Sleep until next update needed
                sleep_time = min(time_to_rt, time_to_da, 60)  # Check at least every minute
                if sleep_time > 0:
                    logger.debug(f"Sleeping for {sleep_time:.1f} seconds")
                    time.sleep(sleep_time)

                # Run updates if needed
                self.run_once()

            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                break
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                traceback.print_exc()
                time.sleep(30)  # Wait before retry

    def stop(self):
        """Stop the streaming service"""
        logger.info("Stopping CAISO Streaming Service...")
        self.running = False


def main():
    """Main entry point"""
    # Create service
    service = CAISOStreamingService()

    # Handle signals
    def signal_handler(sig, frame):
        service.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start streaming
    try:
        service.start()
    except Exception as e:
        logger.error(f"Service crashed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()