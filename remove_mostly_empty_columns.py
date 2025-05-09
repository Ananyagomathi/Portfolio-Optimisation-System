import pandas as pd

# Load the data, ensuring the first row is used as the header
df = pd.read_csv('cleanedfile.csv')

# Count the columns where the header starts with "Adj Close"
adj_close_count = sum(col.startswith('Volume') for col in df.columns)

# Print the result
print(f'Number of columns starting with "Adj Close": {adj_close_count}')
