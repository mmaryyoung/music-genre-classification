import os
import pandas as pd

def get_track_genre_map(filepath):
    return pd.read_csv(filepath, skiprows=[0, 2], header=0, index_col=0, usecols=['Unnamed: 0', 'genre_top'], dtype={0: 'string'})

# Given a mapping between track ID and genre name, and a list of file paths,
# return a list of unique genre names.
def get_all_genres(genre_map, filepaths):
    all_genres = set()
    for f in filepaths:
        try:
            current_genre = genre_map.at[str(int(os.path.basename(f).split('.')[0])), 'genre_top']
            all_genres.add(current_genre)
        except Exception:
            continue
    print("All genres in files: %s" % str(all_genres) )
    return all_genres