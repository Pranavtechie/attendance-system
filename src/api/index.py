from flask import Flask
import requests as req

app = Flask(__name__)

@app.route("/")
def hello_world():
    return {"message": "Hello, World!"}


@app.route("/test", methods=["GET"])
def test():
    r = req.get('https://api.github.com/events')
    
    return r.json()