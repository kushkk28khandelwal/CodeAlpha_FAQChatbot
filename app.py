import json
import nltk
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

app = Flask(__name__)

with open("faqs.json", "r") as f:
    faqs = json.load(f)

questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(questions)

def get_best_answer(user_input):
    user_input = user_input.strip().lower()
    if not user_input:
        return "Please ask a question!"

    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, faq_vectors).flatten()
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score < 0.1:
        return "I'm sorry, I don't have an answer for that. Try asking something about AI, ML, Python, or CodeAlpha!"

    return answers[best_idx]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    response = get_best_answer(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True, port=5002)
