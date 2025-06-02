import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

print("Creating model visualizations...")

# Load test predictions
test = pd.read_csv('test_predictions.csv', parse_dates=['Date'])
trade_history = pd.read_csv('trade_history.csv', parse_dates=['date'])

# Get top traded stock for visualization
top_ticker = test['Ticker'].value_counts().index[0]
stock_data = test[test['Ticker'] == top_ticker]

# Get last month of data (approx 21 trading days)
end_date = stock_data['Date'].max()
start_date = end_date - timedelta(days=30)
recent_data = stock_data[stock_data['Date'] >= start_date].copy()

# Create figure
plt.figure(figsize=(14, 8))

# Plot price history
plt.plot(recent_data['Date'], recent_data['Close'], 'b-o', label='Close Price')

# Highlight predictions
buy_signals = recent_data[recent_data['Prediction'] == 1]
plt.scatter(
    buy_signals['Date'], 
    buy_signals['Close'], 
    color='green', 
    s=100,
    marker='^',
    label='Buy Signal'
)

# Add trade markers
for _, trade in trade_history.iterrows():
    if trade['date'] >= start_date:
        if trade['type'] == 'buy':
            plt.scatter(
                trade['date'], 
                trade['price'], 
                color='green', 
                s=200,
                marker='^',
                edgecolor='black'
            )
            plt.annotate(
                'BUY', 
                (trade['date'], trade['price']),
                textcoords="offset points", 
                xytext=(0,15),
                ha='center',
                fontsize=10,
                fontweight='bold'
            )
        else:
            plt.scatter(
                trade['date'], 
                trade['price'], 
                color='red', 
                s=200,
                marker='v',
                edgecolor='black'
            )
            plt.annotate(
                f"SELL\n{trade['return']:.1%}", 
                (trade['date'], trade['price']),
                textcoords="offset points", 
                xytext=(0,-25),
                ha='center',
                fontsize=9
            )

# Format plot
plt.title(f'{top_ticker} - Price History with Model Signals\n(Green = Buy Recommendation)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Format x-axis dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
plt.gcf().autofmt_xdate()

# Add performance summary
last_trade = trade_history.iloc[-1]
plt.figtext(0.15, 0.02, 
            f"Final Capital: ${last_trade['price'] * last_trade['shares']:.2f} "
            f"| Trades: {len(trade_history)//2}",
            fontsize=12,
            bbox=dict(facecolor='lightgrey', alpha=0.5))

plt.tight_layout()
plt.savefig('model_signals.png')
print("Saved model_signals.png")

# Create prediction summary chart
plt.figure(figsize=(10, 6))
signal_counts = test['Prediction'].value_counts()
plt.pie(signal_counts, 
        labels=['Hold', 'Buy'], 
        colors=['lightcoral', 'lightgreen'],
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.1, 0))
plt.title('Model Signal Distribution')
plt.savefig('signal_distribution.png')
print("Saved signal_distribution.png")