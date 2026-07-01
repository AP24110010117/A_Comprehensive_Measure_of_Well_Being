from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# ----------------------------
# Load Model
# ----------------------------
model = pickle.load(open("HDI.pkl", "rb"))

# ----------------------------
# Load Dataset
# ----------------------------
data = pd.read_csv("../Dataset/HDI.csv")

# Country List
countries = sorted(data["Country"].unique())


# ----------------------------
# Home Page
# ----------------------------
@app.route("/")
def home():
    return render_template("home.html")


# ----------------------------
# Prediction Page
# ----------------------------
@app.route("/predictpage")
def predictpage():
    return render_template(
        "indexnew.html",
        countries=countries
    )


# ----------------------------
# Prediction
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:

        country = request.form["country"]

        life = float(request.form["life"])

        school = float(request.form["school"])

        gni = float(request.form["gni"])

        # Get encoded country value directly from dataset
        country_code = int(
            data.loc[data["Country"] == country].index[0]
        )

        sample = pd.DataFrame(
            [[country_code, life, school, gni]],
            columns=[
                "Country",
                "Life Expectancy at Birth (2021)",
                "Mean Years of Schooling (2021)",
                "Gross National Income Per Capita (2021)"
            ]
        )

        prediction = model.predict(sample)[0]

        # HDI Category
        if prediction >= 0.800:
            message = "Very High Human Development"

        elif prediction >= 0.700:
            message = "High Human Development"

        elif prediction >= 0.550:
            message = "Medium Human Development"

        else:
            message = "Low Human Development"

        return render_template(
            "resultnew.html",
            country=country,
            prediction=round(prediction, 3),
            message=message
        )

    except Exception as e:
        return f"<h2>Error</h2><br>{e}"


# ----------------------------
# Run Flask
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)