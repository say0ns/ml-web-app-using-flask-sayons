import os
from flask import Flask, request, render_template
from pickle import load

with open('../src/modelos-random-forest.pkl', 'rb') as f:
    modelos = load(f)

app = Flask(__name__)


model = modelos['rf_classifier_model']
class_dict = {"0": "No Diabetes",
              "1": "Diabetes"
              }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template('index.html', prediction=None)
    if request.method == "POST":
        val1 = float(request.form["val1"])
        val2 = float(request.form["val2"])
        val3 = float(request.form["val3"])
        val4 = float(request.form["val4"])
        val5 = float(request.form["val5"])
        val6 = float(request.form["val6"])
        val7 = float(request.form["val7"])
        val8 = float(request.form["val8"])
        data = [[val1, val2, val3, val4, val5, val6, val7, val8]]
        prediction = str(model.predict(data)[0])
        pred_class = class_dict[prediction]
        return render_template('index.html', prediction=pred_class)
    return None


if __name__ == '__main__':
    # PORT = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=3000, debug=True)