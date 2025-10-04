#!/usr/bin/env python3
"""
Redis Setup Script

Initializes Redis data structures for CAISO streaming pipeline.
Creates keys, sets TTLs, and loads sample data for testing.

Usage:
    python setup_redis.py
"""

import redis
import json
from datetime import datetime, timedelta

def setup_redis():
    """Set up Redis data structures"""

    # Connect to Redis
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

    print("Setting up Redis data structures...")

    # 1. Clear existing data (optional)
    print("Clearing existing keys...")
    for key in r.scan_iter("caiso:*"):
        r.delete(key)

    # 2. Initialize current prices hash
    print("Initializing current prices...")
    r.hset('caiso:sp15:current', mapping={
        'rt_price': '45.23',
        'rt_timestamp': datetime.utcnow().isoformat(),
        'da_price': '42.10',
        'da_timestamp': datetime.utcnow().isoformat(),
        'last_update': datetime.utcnow().isoformat(),
        'node': 'TH_SP15_GEN-APND',
        'zone': 'SP15'
    })

    # 3. Add sample historical data to stream
    print("Adding sample stream data...")
    for i in range(10):
        timestamp = datetime.utcnow() - timedelta(minutes=5*i)
        price = 40 + (i * 2)  # Sample prices

        r.xadd('caiso:sp15:stream', {
            'timestamp': timestamp.isoformat(),
            'price': price,
            'type': 'RT',
            'node': 'TH_SP15_GEN-APND',
            'zone': 'SP15'
        })

    # 4. Add sample time series data
    print("Adding sample time series data...")
    for i in range(24):  # Last 24 hours
        timestamp = datetime.utcnow() - timedelta(hours=i)
        rt_price = 45 + (i % 10) * 3  # Varying prices
        da_price = 42 + (i % 8) * 2

        # RT prices (5-min intervals)
        for j in range(12):  # 12 five-minute intervals per hour
            ts = timestamp + timedelta(minutes=5*j)
            price_val = rt_price + j * 0.5
            r.zadd('caiso:sp15:rt:5min', {f"{ts.timestamp()}:{price_val}": ts.timestamp()})

        # DA prices (hourly)
        r.zadd('caiso:sp15:da:hourly', {f"{timestamp.timestamp()}:{da_price}": timestamp.timestamp()})

    # 5. Set up pub/sub test
    print("Setting up pub/sub channels...")
    # Channels are created automatically when subscribed to

    # 6. Create index for stream (optional, for faster queries)
    stream_info = r.xinfo_stream('caiso:sp15:stream')
    print(f"Stream created with {stream_info['length']} entries")

    # 7. Display summary
    print("\n" + "="*50)
    print("Redis Setup Complete!")
    print("="*50)

    print("\nData structures created:")
    print("- caiso:sp15:current (Hash) - Current prices")
    print("- caiso:sp15:stream (Stream) - Price update stream")
    print("- caiso:sp15:rt:5min (Sorted Set) - RT price time series")
    print("- caiso:sp15:da:hourly (Sorted Set) - DA price time series")

    print("\nPub/Sub channels:")
    print("- caiso:sp15:updates - Price updates")
    print("- caiso:alerts - Price alerts")

    # Test retrieval
    print("\n" + "="*50)
    print("Testing data retrieval...")
    print("="*50)

    current = r.hgetall('caiso:sp15:current')
    print(f"\nCurrent prices:")
    print(f"  RT: ${current.get('rt_price')}/MWh")
    print(f"  DA: ${current.get('da_price')}/MWh")

    # Get latest from stream
    latest_stream = r.xrevrange('caiso:sp15:stream', count=1)
    if latest_stream:
        print(f"\nLatest stream entry:")
        print(f"  ID: {latest_stream[0][0]}")
        print(f"  Data: {latest_stream[0][1]}")

    # Get recent RT prices
    end_time = datetime.utcnow().timestamp()
    start_time = (datetime.utcnow() - timedelta(hours=1)).timestamp()
    recent_rt = r.zrangebyscore('caiso:sp15:rt:5min', start_time, end_time, withscores=True, start=0, num=5)

    print(f"\nRecent RT prices (last hour, first 5):")
    for item, score in recent_rt:
        parts = item.split(':')
        if len(parts) == 2:
            ts, price = parts
            print(f"  ${float(price):.2f}/MWh at {datetime.fromtimestamp(float(ts)).strftime('%H:%M')}")

    print("\n✓ Redis is ready for streaming!")

    return r

def test_pubsub(r):
    """Test pub/sub functionality"""
    print("\n" + "="*50)
    print("Testing Pub/Sub...")
    print("="*50)

    # Publish test message
    test_message = {
        'timestamp': datetime.utcnow().isoformat(),
        'price': 50.00,
        'type': 'RT',
        'node': 'TH_SP15_GEN-APND',
        'zone': 'SP15'
    }

    subscribers = r.publish('caiso:sp15:updates', json.dumps(test_message))
    print(f"Published test message to {subscribers} subscribers")

    # Publish test alert
    test_alert = {
        'type': 'high_price',
        'threshold': 100,
        'price': 105.50,
        'timestamp': datetime.utcnow().isoformat()
    }

    alert_subscribers = r.publish('caiso:alerts', json.dumps(test_alert))
    print(f"Published test alert to {alert_subscribers} subscribers")

if __name__ == "__main__":
    try:
        r = setup_redis()
        test_pubsub(r)

        print("\n" + "="*50)
        print("Setup complete! You can now run:")
        print("1. python caiso_streaming_service.py  # Start streaming")
        print("2. python websocket_server.py         # Start WebSocket server")
        print("3. Open dashboard.html in browser     # View dashboard")
        print("="*50)

    except redis.ConnectionError:
        print("\n❌ Error: Cannot connect to Redis!")
        print("Please make sure Redis is installed and running:")
        print("  brew install redis  # Install on macOS")
        print("  redis-server        # Start Redis")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()