from flask import Blueprint, session, jsonify
from Backend.db.mongo import client
main_routes = Blueprint("main", __name__)

@main_routes.route("/")
def home():
    return "Backend Running ✅"

@main_routes.route("/test-db")
def test_db():
    try:
        client.admin.command("ping")
        return "MongoDB connected successfully ✅"
    except Exception as e:
        return f"MongoDB failed ❌ {e}"

@main_routes.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        return jsonify({"error": "Login required"}), 401

    return jsonify({"message": "Prediction successful (dummy response)"})
