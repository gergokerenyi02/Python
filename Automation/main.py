import pandas as pd
import numpy as np

# Read HTMl tables from the URL
currentUrl = 'https://en.wikipedia.org/wiki/List_of_The_Simpsons_episodes_(seasons_1%E2%80%9320)'

data = pd.read_html(currentUrl, header=0)

# Extract the first table
firstTable = data[0] #Series overview


# Extract the season and rating columns
seasons = np.array(firstTable['Season'], dtype=str)[1:]
ratings = np.array(firstTable['Rating'], dtype=str)[1:]

# Replace 'TBA' and '—' with 0
seasons = np.where(seasons == '—', '0', seasons)
ratings = np.where(ratings == 'TBA', '0', ratings)
ratings = np.where(ratings == '—', '0', ratings)

# Convert the ratings to float
ratings = np.array(ratings, dtype=np.float32)

# Find the season with the best rating
maxRatedSeason = seasons[(np.max(ratings) == ratings)][0]

print("The following season has the best rating: Season " + maxRatedSeason)

# AVG rating

avg = np.round((np.sum(ratings) / ratings.size), 2)

print(avg)