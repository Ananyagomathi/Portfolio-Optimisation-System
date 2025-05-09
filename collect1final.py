import yfinance as yf
import pandas as pd
# Path to your file
file_path = 'try1.txt'

# List to hold the first elements
tickers = []

# Open the file and read line by line
with open(file_path, 'r') as file:
    for line in file:
        # Strip any leading/trailing whitespace and split by comma
        elements = line.strip().split(',')
        if elements:  # Check if there is at least one element
            tickers.append(elements[0])  # Add the first element to the list

# Printing the results
"""print(first_elements)"""

# Define the ticker symbols for the companies you're interested in
"""tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "FB"]"""

# Fetch the stock price data
def get_stock_data(tickers):
    data = yf.download(tickers, start="2008-01-01", end="2018-01-01")
    return data

# Save the data to CSV
def save_to_csv(data, filename):
    data.to_csv(filename)

# Main function to run the program
def main():
    stock_data = get_stock_data(tickers)
    save_to_csv(stock_data, "stock_prices.csv")

if __name__ == "__main__":
    main()
