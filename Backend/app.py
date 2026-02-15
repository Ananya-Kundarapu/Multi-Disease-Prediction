from flask import Flask
from flask_bcrypt import Bcrypt
from Backend.routes.main import main_routes
from Backend.auth.routes import auth_routes

app = Flask(__name__)
app.secret_key = "super_secret_key_change_this"

bcrypt = Bcrypt(app)

app.register_blueprint(main_routes)
app.register_blueprint(auth_routes)

if __name__ == "__main__":
    app.run(debug=True)