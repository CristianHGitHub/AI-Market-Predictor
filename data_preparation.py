import pandas as pd

print("Preparing data...")
# Load cleaned dataset
df = pd.read_csv('world_stocks_cleaned.csv', parse_dates=['Date'])

# Handle missing values (fixed deprecated method)
df = df.ffill()  # Forward fill
df = df.dropna()  # Remove any remaining NA

# Sort data chronologically per ticker
df.sort_values(['Ticker', 'Date'], inplace=True)

# Save prepared data
df.to_csv('prepared_data.csv', index=False)
print("Saved prepared data as 'prepared_data.csv'")