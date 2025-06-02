import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report
import joblib
import os
import matplotlib.pyplot as plt

print("Training models...")
# Check if files exist
if not os.path.exists('train_data.csv') or not os.path.exists('test_data.csv'):
    print("Error: Run feature_engineering.py first to create train/test data")
    exit()

# Load data with date parsing
train = pd.read_csv('train_data.csv', parse_dates=['Date'])
test = pd.read_csv('test_data.csv', parse_dates=['Date'])

# Feature selection
features = ['MA_7', 'Volatility', 'RSI', 'Volume_Change']
X_train = train[features]
y_train = train['Target']
X_test = test[features]
y_test = test['Target']

# Validate data - check for infinite or NaN values
def validate_data(df, name):
    print(f"\nValidating {name} data...")
    print(f"Rows: {len(df)}")
    print("Missing values:")
    print(df.isnull().sum())
    print("Infinite values:")
    print(np.isinf(df).sum())
    print("Data types:")
    print(df.dtypes)

validate_data(X_train, "training features")
validate_data(X_test, "test features")

# Replace any remaining inf/-inf with NaN and drop
X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
y_train = y_train.loc[X_train.index]
X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna()
y_test = y_test.loc[X_test.index]

# Initialize models with optimized parameters
models = {
    "XGBoost": XGBClassifier(
        objective='binary:logistic',
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        tree_method='hist'
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    # Print report
    print(f"{name} Performance:")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}")
    print(classification_report(y_test, preds))
    
    # Save results
    results[name] = {
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'model': model,
        'predictions': preds
    }

# Save best model based on AUC
best_model_name = max(results, key=lambda x: results[x]['auc'])
best_model = results[best_model_name]['model']
joblib.dump(best_model, 'best_model.pkl')
print(f"\nSaved best model ({best_model_name}) as 'best_model.pkl'")

# REALISTIC BACKTEST SIMULATION (FIXED)
def backtest(model, test_data, predictions):
    capital = 10000
    position = 0
    trade_active = False
    buy_price = 0
    trades = []
    
    # Clean data for backtesting
    test_data = test_data.copy()
    test_data['Prediction'] = predictions
    
    # Sort by date and ticker
    test_data.sort_values(['Date', 'Ticker'], inplace=True)
    
    # Get most traded stock for backtest
    top_ticker = test_data['Ticker'].value_counts().index[0]
    stock_data = test_data[test_data['Ticker'] == top_ticker].copy()
    stock_data.sort_values('Date', inplace=True)
    
    # Simulate weekly trades
    for i in range(len(stock_data)):
        current = stock_data.iloc[i]
        
        # Entry signal (buy)
        if not trade_active and current['Prediction'] == 1:
            buy_price = current['Close']
            shares = capital / buy_price
            trade_active = True
            entry_date = current['Date']
            trades.append({
                'type': 'buy',
                'date': current['Date'],
                'price': buy_price,
                'shares': shares
            })
        
        # Exit after 5 days
        if trade_active and (current['Date'] - entry_date).days >= 5:
            sell_price = current['Close']
            capital = shares * sell_price
            trade_active = False
            trades.append({
                'type': 'sell',
                'date': current['Date'],
                'price': sell_price,
                'return': (sell_price - buy_price) / buy_price
            })
    
    # Liquidate final position
    if trade_active:
        sell_price = stock_data.iloc[-1]['Close']
        capital = shares * sell_price
        trades.append({
            'type': 'sell',
            'date': stock_data.iloc[-1]['Date'],
            'price': sell_price,
            'return': (sell_price - buy_price) / buy_price
        })
    
    # Save trade history for visualization
    trade_history = pd.DataFrame(trades)
    trade_history.to_csv('trade_history.csv', index=False)
    
    return capital, top_ticker

# Run backtest with predictions
final_capital, top_ticker = backtest(
    best_model, 
    test.copy(), 
    results[best_model_name]['predictions']
)
print(f"Backtest result - Final capital: ${final_capital:.2f}")
print(f"Backtest performed on: {top_ticker}")

# Save predictions for visualization
test['Prediction'] = results[best_model_name]['predictions']
test.to_csv('test_predictions.csv', index=False)
print("Saved test predictions to 'test_predictions.csv'")