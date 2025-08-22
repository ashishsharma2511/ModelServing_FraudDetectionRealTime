from kafka import KafkaProducer
import json
import time
import random


producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

merchants = [f"M{i:03d}" for i in range(1, 21)]
categories = ['grocery', 'electronics', 'clothing', 'food', 'jewelry', 'travel', 'luxury']
devices = ['mobile', 'desktop', 'tablet']
locations = ['US', 'CA', 'UK']
txn_id=1
def generate_transaction(txn_id=None):
    return {
       
        "transaction_id": txn_id,
        "user_id": random.randint(1000, 2000),
        "amount": round(random.expovariate(1/100), 2),
        "merchant": random.choice(merchants),
        "category": random.choice(categories),
        "time_of_day": random.randint(0, 23),
        "device_type": random.choice(devices),
        "location": random.choice(locations)
    }
        
    
print("Producing messages to 'transactions' topic...")

while True:
    txn = generate_transaction(txn_id)
    txn_id += 1
    producer.send('evtransactions', value=txn)
    print(f" Sent: {txn}")
    time.sleep(2)  # 1 event per 2 second