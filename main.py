import pandas as pd

music_df = pd.read_csv('./data/high_popularity_spotify_data.csv')

print(music_df.columns)
print(music_df.track_name.head())
print(music_df.describe())