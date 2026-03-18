from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 👇 IMPORTANT CHANGE (connect frontend)
app = Flask(__name__, static_folder='frontend')
CORS(app)

# Load dataset
df = pd.read_csv("data/spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ["label", "message"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"])

model = MultinomialNB()
model.fit(X, df["label"])

# 🔥 Home route (opens login page)
@app.route("/")
def home():
    return send_from_directory("frontend", "login.html")

# 🔥 Serve all frontend files (dashboard, css, etc.)
@app.route("/<path:path>")
def serve_files(path):
    return send_from_directory("frontend", path)

# API route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "")

    vec = vectorizer.transform([message])
    prediction = model.predict(vec)

    result = "SPAM" if prediction[0] == 1 else "NOT SPAM"

    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(debug=True)