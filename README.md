# 📈 MSFT Stock Price Prediction System

A full-stack AI application that predicts Microsoft (MSFT) stock movements using Machine Learning, Deep Learning, and Natural Language Processing.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![React](https://img.shields.io/badge/React-18.0+-61DAFB.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-000000.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Features

- **Trend Prediction**: XGBoost classifier predicts UP/DOWN movements
- **Price Prediction**: LSTM neural network forecasts next-day closing price
- **Sentiment Analysis**: VADER NLP analyzes news sentiment from Yahoo Finance
- **Trading Signals**: Generates BUY/SELL/HOLD recommendations
- **Real-time Data**: Fetches live stock data and news (no API key required!)
- **Modern UI**: React-based responsive web interface

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   React UI  │ ───> │  Flask API   │ ───> │  ML/DL Models   │
│ (Port 3000) │      │ (Port 5000)  │      │  - XGBoost      │
└─────────────┘      └──────────────┘      │  - LSTM         │
                                            │  - VADER NLP    │
                                            └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Node.js 14 or higher
- npm or yarn

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/msft-stock-prediction.git
cd msft-stock-prediction
```

**2. Set up Python environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**3. Set up Frontend**
```bash
cd frontend
npm install
cd ..
```

**4. Train the models** (First time only)
```bash
# Download and process data
python backend/data_collection.py
python backend/data_processing.py

# Train models
python backend/train_trend_model.py
python backend/train_price_lstm.py

# Setup NLP
python backend/nltk_setup.py
```

### Running the Application

**Terminal 1 - Start Backend:**
```bash
python backend/api.py
```
Backend runs at: `http://127.0.0.1:5000`

**Terminal 2 - Start Frontend:**
```bash
cd frontend
npm start
```
Frontend opens at: `http://localhost:3000`

## 📊 How It Works

### 1. Data Collection
- Fetches 6 months of MSFT historical data from Yahoo Finance
- Downloads latest news headlines (free, no API key needed!)

### 2. Feature Engineering
Computes 8 technical indicators:
- **Moving Averages**: MA10, MA20, MA50
- **Momentum**: RSI (Relative Strength Index)
- **Trend**: MACD (Moving Average Convergence Divergence)
- **Volatility**: Return, Momentum, Standard Deviation

### 3. AI Predictions

**XGBoost Trend Model**
- Binary classification: UP (1) or DOWN (0)
- 8 features, ~48% accuracy
- Trained on 4,000+ historical data points

**LSTM Price Model**
- Sequence-to-sequence prediction
- 30-day lookback window
- 2 LSTM layers with 64 units each
- Dropout regularization (0.2)

**VADER Sentiment Analysis**
- Pre-trained NLP model
- Analyzes news headlines
- Outputs: Positive, Negative, or Neutral

### 4. Trading Signal Logic

```python
if trend == UP and predicted_price > current_price and sentiment == Positive:
    signal = "BUY"
elif trend == DOWN and sentiment == Negative:
    signal = "SELL"
else:
    signal = "HOLD"
```

## 📁 Project Structure

```
msft-stock-prediction/
├── backend/                    # Python backend
│   ├── api.py                 # Flask REST API
│   ├── trading_signal.py      # Main prediction engine
│   ├── data_collection.py     # Data fetching
│   ├── data_processing.py     # Feature engineering
│   ├── train_trend_model.py   # XGBoost training
│   ├── train_price_lstm.py    # LSTM training
│   ├── news_fetcher.py        # News scraper
│   ├── sentiment_model.py     # VADER NLP
│   ├── nltk_setup.py          # NLP setup
│   ├── models/                # Trained models
│   │   ├── trend_model.pkl
│   │   ├── price_model.h5
│   │   └── scaler.pkl
│   └── data/                  # Stock data
│       ├── msft_raw.csv
│       └── msft_processed.csv
├── frontend/                   # React frontend
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js             # Main component
│   │   ├── index.js           # Entry point
│   │   └── index.css          # Styles
│   └── package.json
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                   # This file
```

## 🔧 API Documentation

### GET /predict

Returns stock prediction and trading signal.

**Response:**
```json
{
  "current_price": 424.82,
  "predicted_price": 393.27,
  "trend": "DOWN",
  "sentiment": "Neutral",
  "signal": "HOLD"
}
```

## 🧪 Model Performance

| Model | Metric | Value |
|-------|--------|-------|
| XGBoost | Accuracy | 48.5% |
| LSTM | MSE Loss | 0.0036 |
| VADER | Pre-trained | N/A |

**Note**: 48% accuracy is normal for stock prediction due to market randomness. The system combines multiple signals for better decision-making.

## 🔄 Retraining Models

To retrain with fresh data:

```bash
# Step 1: Collect new data
python backend/data_collection.py

# Step 2: Process features
python backend/data_processing.py

# Step 3: Train models
python backend/train_trend_model.py
python backend/train_price_lstm.py
```

**Recommended retraining frequency:**
- **Weekly**: For active trading
- **Monthly**: For long-term investing
- **After major events**: Earnings reports, market crashes

## 🛠️ Technologies Used

### Backend
- **Python 3.10+**
- **Flask** - REST API framework
- **XGBoost** - Gradient boosting for classification
- **TensorFlow/Keras** - Deep learning framework
- **NLTK VADER** - Sentiment analysis
- **yfinance** - Yahoo Finance data (free!)
- **ta** - Technical analysis library
- **pandas** - Data manipulation
- **scikit-learn** - ML utilities

### Frontend
- **React 18** - UI framework
- **JavaScript ES6+**
- **Fetch API** - HTTP requests

## ❓ FAQ

**Q: Do I need an API key for stock data or news?**  
A: No! We use `yfinance` which scrapes Yahoo Finance's public website. It's completely free.

**Q: Why is the accuracy only 48%?**  
A: Stock markets are inherently unpredictable. Even 51% accuracy can be profitable when combined with proper risk management. Our system uses 3 different models to improve decision-making.

**Q: Can I use this for other stocks?**  
A: Currently optimized for MSFT. To support other tickers, modify `trading_signal.py` line 27 and retrain models with new data.

**Q: Is this safe for real trading?**  
A: **NO!** This is for educational purposes only. Never trade real money without proper risk management and professional financial advice.

**Q: How does news fetching work without an API?**  
A: The `yfinance` library scrapes Yahoo Finance's publicly available news section. No authentication required.

## 🐛 Troubleshooting

### Backend won't start
- Make sure virtual environment is activated
- Check if port 5000 is available
- Verify all dependencies: `pip list`

### Frontend won't start
- Ensure Node.js is installed: `node --version`
- Delete `node_modules` and run `npm install` again
- Check if port 3000 is available

### Models not found
- Run training scripts in order (see Quick Start step 4)
- Verify `backend/models/` folder exists

### Import errors
- Activate virtual environment
- Reinstall: `pip install -r requirements.txt`

## ⚠️ Disclaimer

**This project is for educational purposes only.**

- Not financial advice
- Do not use for actual trading without proper risk management
- Past performance does not guarantee future results
- Consult a licensed financial advisor before making investment decisions
- The creators are not responsible for any financial losses

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance data
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting library
- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework
- [NLTK](https://www.nltk.org/) - Natural language processing
- [React](https://reactjs.org/) - Frontend framework
- [Flask](https://flask.palletsprojects.com/) - Backend framework

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/yourusername/msft-stock-prediction](https://github.com/yourusername/msft-stock-prediction)

---

**Made with ❤️ for learning AI and Full-Stack Development**
