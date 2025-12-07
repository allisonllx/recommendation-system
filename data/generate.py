import random
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

# set seed for reproducibility

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

NUM_USERS = 500
NUM_ITEMS = 200

users = [f"user_{i}" for i in range(NUM_USERS)]
items = [f"item_{i}" for i in range(NUM_ITEMS)]
user_profiles = [{"user_id": u, "age": fake.random_int(18,50), "region": fake.city()} for u in users]

INTERACTIONS_PER_USER = 50
interaction_types = ["click", "like", "comment", "share"]
data = []

for user in users:
    start_time = datetime(2024, 1, 1)
    for _ in range(INTERACTIONS_PER_USER):
        item = random.choice(items)
        inter_type = random.choices(interaction_types, weights=[0.6, 0.25, 0.1, 0.05])[0]
        timestamp = start_time + timedelta(minutes=random.randint(1, 365*24*60 - 1)) # 1-year duration
        data.append({
            "user_id": user,
            "item_id": item,
            "interaction_type": inter_type,
            "timestamp": timestamp
        })

df = pd.DataFrame(data)

df = df.sort_values(by=["user_id","timestamp"])

# map categorical columns to integer IDs
user2idx = {user: idx for idx ,user in enumerate(df.user_id.unique())}
item2idx = {item: idx for idx, item in enumerate(df.item_id.unique())}
inter_type2idx = {t:i for i,t in enumerate(df.interaction_type.unique())}

df["user_idx"] = df["user_id"].map(user2idx)
df["item_idx"] = df["item_id"].map(item2idx)
df["interaction_idx"] = df["interaction_type"].map(inter_type2idx)

# group interactions into sequences per user
user_sequences = df.groupby("user_idx").apply(lambda x: list(zip(x.item_idx, x.interaction_idx, x.timestamp)))

# Save Dataset

df.to_csv("mock_interactions.csv", index=False)
print("Mock dataset saved as mock_interactions.csv")