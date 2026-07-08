import os
import joblib

from flask import Flask, render_template, request
from utils.extract_text import extract_text
from model.preprocess import clean_text

app = Flask(__name__)

#Configure Upload Folder
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
app.config["UPLOAD_FOLDER"]= UPLOAD_FOLDER

#Load the Saved Model
MODEL_PATH = os.path.join(BASE_DIR,"saved_model", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "saved_model","vectorizer.pkl")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

#Create the homepage route
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    if "resume" not in request.files:
        return "No file uploaded."
    file = request.files["resume"]
    
    if file.filename == "":
        return "No file selected."
    
    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )
    file.save(file_path)
    raw_text =  extract_text(file_path)
    cleaned_text = clean_text(raw_text)
    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)[0]
    os.remove(file_path)

    return render_template(
        "result.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)