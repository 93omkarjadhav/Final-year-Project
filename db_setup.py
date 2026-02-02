import pandas as pd
from sqlalchemy import create_engine
from pymongo import MongoClient

# Dummy Data
data = {
    'product_name': ['iPhone 15', 'Samsung S24', 'Sony Headphones', 'Beauty Cream', 'Laptop'],
    'category': ['Electronics', 'Electronics', 'Electronics', 'Beauty', 'Electronics'],
    'price': [799, 899, 299, 45, 1200],
    'stock': [50, 40, 100, 200, 15]
}
df = pd.DataFrame(data)

def setup_all():
    # 1. PostgreSQL/Oracle (SQL)
    pg_engine = create_engine("postgresql+psycopg2://user:pass@localhost:5432/retail_db")
    df.to_sql('products', pg_engine, if_exists='replace', index=False)
    
    # 2. MongoDB (NoSQL)
    mongo_client = MongoClient("mongodb://localhost:27017/")
    db = mongo_client["retail_sales_nosql"]
    db["products"].insert_many(df.to_dict('records'))
    print("✅ All databases updated with dummy data!")

if __name__ == "__main__":
    setup_all()