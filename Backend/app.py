# app.py
from flask import Flask
from api.routers import api

app = Flask(__name__)
app.register_blueprint(api, url_prefix="/api")

@app.route('/')
def index():
    return "Hello World"

if __name__ == '__main__':
    app.run(debug=True)
