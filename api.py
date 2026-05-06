import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS   # ✅ ADD THIS

app = Flask(__name__)
CORS(app)   # ✅ VERY IMPORTANT

# model load
model = pickle.load(open('model.pkl', 'rb'))

# feature columns
columns = ['age','weight','height','exercise','sleep',
           'sugar_intake','smoking','married','profession','bmi']


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['input']
        print("Received:", data)

        df = pd.DataFrame([data], columns=columns)

        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0]
        confidence = float(max(proba))

        return jsonify({
            "prediction": str(pred),   # ✅ safe convert
            "confidence": confidence
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)