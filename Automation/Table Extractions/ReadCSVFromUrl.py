import pandas as pd

# Set URL
url = 'https://www.football-data.co.uk/mmz4281/2122/E0.csv'

# Read CSV from URL
df = pd.read_csv(url)

# Rename columns
df.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'}, inplace=True)

print(df.head())