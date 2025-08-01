import os
from flask import Flask, request, render_template
from pickle import load

with open('../src/modelos-random-forest.pkl', 'rb') as f:
    modelos = load(f)

app = Flask(__name__)

model = modelos['rf_classifier_model']
class_dict = {
    0: "No Diabetes",
    1: "Diabetes"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template('index.html', prediction=None)
    if request.method == "POST":
        try:
            data = [
                [float(request.form[f"val{i}"]) for i in range(1, 9)]
            ]
            prediction = int(model.predict(data)[0])
            pred_class = class_dict[prediction]
            return render_template('index.html', prediction=pred_class)
        except Exception as e:
            return f"Error: {e}", 500
    return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
