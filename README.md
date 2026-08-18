# PropSense — House Price Prediction Website

A full-stack ML web application built with Flask + scikit-learn.

## How to Run

### 1. Install dependencies
```
pip install flask numpy scikit-learn
```

### 2. Start the server
```
python app.py
```

### 3. Open your browser
Go to: http://127.0.0.1:5000

## Project Structure
```
house_price_app/
├── app.py              ← Flask backend + ML models
├── requirements.txt    ← Python packages needed
├── README.md           ← This file
└── templates/
    └── index.html      ← Frontend website
```

## Features
- Live price prediction using 3 ML models
- Gradient Boosting, Random Forest, Ridge Regression
- Feature importance chart
- Model performance comparison
- Works on desktop and mobile
