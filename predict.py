import os
import model
from flask import Flask, request

RUN_ID = os.getenv('RUN_ID')

model_service = model.init(run_id=RUN_ID)

app = Flask('injury-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    passenger = request.get_json()

    result = model_service.request_handler(passenger)

    return result

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)