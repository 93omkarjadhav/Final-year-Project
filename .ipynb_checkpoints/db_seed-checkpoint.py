import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def seed_mongodb():
    # Uses the URI you provided in your .env
    uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME")
    
    client = MongoClient(uri)
    db = client[db_name]
    collection = db["products"] # Your collection name

    # Dummy data representing retail products
    dummy_products = [
        {"name": "UltraBook Pro", "category": "Electronics", "price": 1500, "stock": 5},
        {"name": "Wireless Mouse", "category": "Electronics", "price": 25, "stock": 50},
        {"name": "Facial Cleanser", "category": "Beauty", "price": 15, "stock": 100},
        {"name": "Organic Tea", "category": "Food", "price": 10, "stock": 200}
    ]

    collection.delete_many({}) # Clears existing data
    collection.insert_many(dummy_products)
    print(f"✅ Successfully added {len(dummy_products)} records to MongoDB!")

if __name__ == "__main__":
    seed_mongodb()