from flask import Flask
import requests as req
import time

app = Flask(__name__)

@app.route("/")
def hello_world():
    return {"message": "Hello, World!"}


@app.route("/test", methods=["GET"])
def test():
    r = req.get('https://api.github.com/events')
    
    return r.json()


# new request to return the current timestamp in milliseconds
@app.route("/timestamp", methods=["GET"])
def timestamp():
    return str(int(time.time() * 1000)) 