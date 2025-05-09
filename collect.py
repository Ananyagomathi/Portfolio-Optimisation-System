import yfinance as yf
# Create a Ticker object for the stock you want (e.g., Apple - AAPL)
ticker = yf.Ticker("AAPL")

# Get historical market data
historical_data = ticker.history(period="1y")  # Adjust period as needed (1mo, 1d, etc.)

# Print the historical data
print(historical_data)
historical_data.to_csv('Apple.csv')