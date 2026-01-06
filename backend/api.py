from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Load model của bạn (ví dụ TF-IDF + cosine)
with open("trained_models/ngram_model.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("trained_models/retrieval_model.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    # xử lý prediction
    # ...
    result = "kết quả mô hình ở đây"

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
