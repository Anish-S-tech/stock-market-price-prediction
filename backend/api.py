from flask import Flask, jsonify
from flask_cors import CORS

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.trading_signal import TradingSignalEngine

app = Flask(__name__)
CORS(app)

engine = TradingSignalEngine()


@app.route("/predict", methods=["GET"])
def predict():

    result = engine.generate_signal()

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
