from flask import Blueprint, request, jsonify, session
from flask_bcrypt import Bcrypt
from Backend.db.mongo import db

auth_routes = Blueprint("auth", __name__)
bcrypt = Bcrypt()

users_collection = db.users

@auth_routes.route("/signup", methods=["POST"])
def signup():
    data = request.json

    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    if not name or not email or not password:
        return jsonify({"error": "All fields are required"}), 400

    if users_collection.find_one({"email": email}):
        return jsonify({"error": "User already exists"}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")

    users_collection.insert_one({
        "name": name,
        "email": email,
        "password": hashed_password
    })

    return jsonify({"message": "User created successfully"}), 201

@auth_routes.route("/login", methods=["POST"])
def login():
    data = request.json

    email = data.get("email")
    password = data.get("password")

    user = users_collection.find_one({"email": email})

    if not user:
        return jsonify({"error": "Invalid credentials"}), 401

    if not bcrypt.check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid credentials"}), 401

    session["user_id"] = str(user["_id"])
    session["user_name"] = user["name"]

    return jsonify({"message": "Login successful"}), 200

@auth_routes.route("/logout")
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully"})

@auth_routes.route("/check-auth")
def check_auth():
    if "user_id" in session:
        return jsonify({
            "authenticated": True,
            "user": session.get("user_name")
        })
    return jsonify({"authenticated": False})
