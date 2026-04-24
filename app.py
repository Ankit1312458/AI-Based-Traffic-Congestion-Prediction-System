from flask import Flask, render_template, request
from models.predict import load_model, predict_traffic, classify_congestion

app = Flask(__name__)

model = load_model()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    hour = int(request.form['hour'])
    day = int(request.form['day'])
    lag1 = float(request.form['lag1'])
    lag2 = float(request.form['lag2'])

    prediction = predict_traffic(model, hour, day, lag1, lag2)
    congestion = classify_congestion(prediction)

    return render_template(
        'dashboard.html',
        prediction=round(prediction, 2),
        congestion=congestion
    )


if __name__ == "__main__":
    app.run(debug=True)