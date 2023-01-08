from collections import defaultdict
from scipy.spatial.distance import cdist
import difflib
number_cols = ['valence','year','acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']
