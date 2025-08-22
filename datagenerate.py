import pandas as pd
import numpy as np

np.random.seed(42)

n = 10000
data = {
    'transaction_id': np.arange(1, n+1),
    'user_id': np.random.randint(1000, 2000, size=n),
    'amount': np.round(np.random.exponential(scale=100, size=n), 2),
    'merchant': np.random.choice([f"M{i:03d}" for i in range(1, 21)], size=n),
    'category': np.random.choice(['grocery', 'electronics', 'clothing', 'food', 'jewelry', 'travel', 'luxury'], size=n),
    'time_of_day': np.random.randint(0,24,size=n),
    'device_type': np.random.choice(['mobile','desktop','tablet'], size=n),
    'location': np.random.choice(['US','CA','UK'], size=n),
    'is_fraud': np.random.choice([0,1], size=n, p=[0.97,0.03])  # 3% fraud
}

df = pd.DataFrame(data)
df.to_csv("sample_transactions.csv", index=False)
print("CSV file saved as sample_transactions.csv")
