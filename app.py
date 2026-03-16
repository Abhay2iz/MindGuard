from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Positive and negative word lists
positive_words = [
    "happy","good","great","excited","joy","relaxed",
    "awesome","fantastic","amazing","wonderful","fine",
    "peaceful","satisfied","glad"
]

negative_words = [
    "stress","tired","overwhelmed","anxious","pressure",
    "depressed","sad","panic","worried","burnout"
]

# Advice generator
def generate_advice(stress_score):

    if stress_score < 30:
        return "You seem to be doing well. Maintain a balanced routine and keep a positive mindset."

    elif stress_score < 60:
        return "You might be experiencing some pressure. Try taking a short break, stretching, or going for a walk."

    else:
        return "You may be under significant stress. Consider deep breathing, relaxation, or talking to someone you trust."


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    text = request.form["text"]

    # Vectorize input text
    text_vec = vectorizer.transform([text])

    # Get prediction probability
    probability = model.predict_proba(text_vec)

    text_lower = text.lower()
    stress_prob = probability[0][1]

    # Hybrid logic (ML + rule based)
    if any(word in text_lower for word in positive_words):
        stress_score = int(stress_prob * 10)

    elif any(word in text_lower for word in negative_words):
        stress_score = int(stress_prob * 100)

    else:
        stress_score = int(stress_prob * 60)

    # Generate advice
    advice = generate_advice(stress_score)

    # Stress level text
    if stress_score < 30:
        result = "Low Stress Detected"
    elif stress_score < 60:
        result = "Moderate Stress Level"
    else:
        result = "High Stress Level"

    return render_template(
        "result.html",
        prediction=result,
        score=stress_score,
        advice=advice
    )


if __name__ == "__main__":
    app.run(debug=True)