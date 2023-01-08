
from collections import defaultdict
from scipy.spatial.distance import cdist
import difflib
number_cols = ['valence','year','acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

def get_song_data(song,spotify_data)
'''gets all the song details for the selected songs. Song takes in input as key value pairs dictionary in form of name and year'''
try:
   song_data = spotify_data[(spotify_data['name'] == song['name']) & (spotify_data['year'] == song['year'])].iloc[0]
    
    return song_data