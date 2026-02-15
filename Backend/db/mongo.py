# Backend/db/mongo.py
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000,
    tlsCAFile=certifi.where()
)

db = client.medpredict
users_collection = db.users
