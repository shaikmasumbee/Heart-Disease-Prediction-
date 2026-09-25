from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("Heart_Disease_Model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaled_model.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None

    if request.method == "POST":

        try:
            # Collect form inputs
            features = [
                float(request.form["age"]),
                float(request.form["sex"]),
                float(request.form["cp"]),
                float(request.form["trestbps"]),
                float(request.form["chol"]),
                float(request.form["fbs"]),
                float(request.form["restecg"]),
                float(request.form["thalach"]),
                float(request.form["exang"]),
                float(request.form["oldpeak"]),
                float(request.form["slope"]),
                float(request.form["ca"]),
                float(request.form["thal"])
            ]

            # Convert inputs into NumPy array
            features_array = np.array([features])

            # Scale the features
            features_scaled = scaler.transform(features_array)

            # Make prediction
            prediction = model.predict(features_scaled)[0]

            if prediction == 0:
                return render_template(
                    "index.html",
                    prediction="No Heart Disease"
                )
            else:
                return render_template(
                    "index.html",
                    prediction="Heart Disease Detected"
                )

        except Exception as e:

            prediction = f"Error: {str(e)}"

            return render_template(
                "index.html",
                prediction=prediction
            )

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)