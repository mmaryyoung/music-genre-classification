import pandas as pd

def get_track_genre_map(filepath):
    return pd.read_csv(filepath, skiprows=[0, 2], header=0, index_col=0, usecols=['Unnamed: 0', 'genre_top'], dtype={0: 'string'})