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


print("The average rating is: " + str(avg))

# Get dates

dates = np.array(firstTable['Originally aired'], dtype=str)[1:]
dates = [date.replace('\xa0', ' ') for date in dates]


dates = pd.to_datetime(dates)
dates = np.array(dates, dtype=np.datetime64)





# Get the best rated season between two dates

def getDateIdxs(fromDate, toDate):
    return np.where(np.logical_and(dates >= np.datetime64(fromDate), dates <= np.datetime64(toDate)))[0]


filtered_date_idxs = getDateIdxs('2000-01-01', '2012-12-31')
filtered_ratings = ratings[filtered_date_idxs]

maxRatedSeason_time = seasons[(np.max(filtered_ratings) == ratings)][0]


print("The following season has the best rating between 2000 and 2012: Season " + maxRatedSeason_time)