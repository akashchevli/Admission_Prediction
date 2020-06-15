from flask import Flask, render_template, request
import pickle
import numpy as np

filename = 'admission_prediction.pkl'
model = pickle.load(open(filename, 'rb'))
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    try:
        if request.method == 'POST':
            gre = int(request.form['grescore'])
            toefl = int(request.form['toeflscore'])
            uni_rating = float(request.form['unirating'])
            lor = float(request.form['lor'])
            cgpa = float(request.form['cgpa'])
            research = int(request.form['research'])

            features = np.array([[gre,toefl,uni_rating,lor,cgpa,research]])
            prediction = model.predict(features)
            output = round(prediction[0]*100, 3)

            return render_template('result.html', prediction_text=output)
        else:
            return render_template('index.html')

    except Exception as e:
        return "Something went wrong"


if __name__ == '__main__':
    app.run(debug=True)


