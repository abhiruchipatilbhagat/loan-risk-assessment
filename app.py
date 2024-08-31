from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("loan_risk_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get data from form
    data = [float(x) for x in request.form.values()]
    features = np.array(data).reshape(1, -1)

    # Predict risk probability
    risk_probability = model.predict_proba(features)[0][
        1
    ]  # Get probability of class '1'

    return render_template(
        "index.html", prediction_text=f"Risk Probability: {risk_probability:.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)
