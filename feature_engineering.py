import pandas as pd
import numpy as np

print("Engineering features...")
# Load prepared data
df = pd.read_csv('prepared_data.csv', parse_dates=['Date'])

# Calculate essential features
df['Returns'] = df.groupby('Ticker')['Close'].pct_change()
df['MA_7'] = df.groupby('Ticker')['Close'].rolling(7).mean().reset_index(level=0, drop=True)
df['Volatility'] = df.groupby('Ticker')['Returns'].rolling(14).std().reset_index(level=0, drop=True)
df['Volume_Change'] = df.groupby('Ticker')['Volume'].pct_change()

# Calculate RSI (fixed version)
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    # Handle division by zero and NaN values
    avg_loss = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = df.groupby('Ticker')['Close'].transform(compute_rsi)

# Create target variable: Next week's price movement (5 trading days)
df['Future_Close'] = df.groupby('Ticker')['Close'].shift(-5)
df['Target'] = (df['Future_Close'] > df['Close']).astype(int)

# Handle infinite values and missing data
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Split data (time-based)
train = df[df['Date'] < '2020-01-01']
test = df[df['Date'] >= '2020-01-01']

# Save engineered data
train.to_csv('train_data.csv', index=False)
test.to_csv('test_data.csv', index=False)
print("Feature engineering complete. Train/test data saved.")
print(f"Train samples: {len(train)}, Test samples: {len(test)}")

# Need to update this file
