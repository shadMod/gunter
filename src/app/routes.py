from flask import render_template, request
from app import app
from model.predict import chatbot_response
from model.train import add_training_data, retrain_model


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["GET"])
def chatbot_reply():
    user_input = request.args.get("msg")
    response = chatbot_response(user_input)
    return response


@app.route("/feedback", methods=["POST"])
def feedback():
    user_input = request.form.get("msg")
    correct_response = request.form.get("correct_response")
    add_training_data(user_input, correct_response)
    retrain_model()
    return "Thank you for your feedback!"
