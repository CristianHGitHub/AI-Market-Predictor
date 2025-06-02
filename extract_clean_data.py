import pandas as pd
import zipfile

print("Extracting and cleaning data...")
with zipfile.ZipFile('archive.zip') as z:
    with z.open('World-Stock-Prices-Dataset.csv') as f:
        df = pd.read_csv(f)
        
        # Convert and clean date column
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df['Date'] = df['Date'].dt.tz_convert(None)
        
        # Select essential columns
        columns_to_keep = [
            'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Brand_Name', 'Ticker', 'Industry_Tag', 'Country',
            'Dividends', 'Stock Splits'
        ]
        df = df[columns_to_keep]
        
        # Clean missing values
        df.dropna(subset=['Date', 'Close', 'Ticker'], inplace=True)
        
        # Normalize text columns
        df['Brand_Name'] = df['Brand_Name'].str.title()
        df['Industry_Tag'] = df['Industry_Tag'].str.lower()
        df['Country'] = df['Country'].str.lower()
        
        # Filter date range
        df = df[(df['Date'] >= '2000-01-01') & (df['Date'] <= '2025-12-31')]
        
        # Save cleaned data
        df.to_csv('world_stocks_cleaned.csv', index=False)
        print("Saved cleaned data as 'world_stocks_cleaned.csv'")