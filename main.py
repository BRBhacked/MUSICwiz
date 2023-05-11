#This code implements spotipy to implement the data from spotify on python
#And create a user freindly way to give a music reccomendation 

#Installing neccesaary python libraries to implement the program 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spotipy
import os
import sklearn

#Reading the dataset files
spotify_data = pd.read_csv('./data/data.csv.zip')
genre_data = pd.read_csv('./data/data_by_genres.csv')
data_by_year = pd.read_csv('./data/data_by_year.csv')

#We are using plotly to visualise (Using a line graph) the values of different audios.
import plotly.express as px 
sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
fig = px.line(data_by_year, x='year', y=sound_features)
fig.show()

#We are using plotly to visualise (Using a graph) the values of different generes.
top10_genres = genre_data.nlargest(10, 'popularity')
fig = px.bar(top10_genres, x='genres', y=['valence', 'energy', 'danceability', 'acousticness'], barmode='group')
fig.show()

X = genre_data.select_dtypes(np.number)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#Implementing kmeans using pipelines
cluster_pipeline = Pipeline([("scaler",StandardScaler()),("kmeans",KMeans(n_clusters = 10))])
cluster_pipeline.fit(X)
genre_data["cluster"] = cluster_pipeline.predict(X)

#Visualising the pipelines implementation of genere clusters into  a two dimensional space 
from sklearn.manifold import TSNE

#Initialising a new dataset with a two dimensional pipeline.
tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=2))])
genre_embedding = tsne_pipeline.fit_transform(X)

#Creating a panda object to implement a 2-dimensional labeled data structure.
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)

#Updating values of generes and clusters from the kmeans implementation rather than from the 2D implementaion.
projection['genres'] = genre_data['genres']
projection['cluster'] = genre_data['cluster']

#Using plotply function to plot the 2D projection created above.
fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])
fig.show()

#Clustering songs with Kmeans

song_cluster_pipeline = Pipeline([('scaler',StandardScaler()),('kmeans',KMeans(n_clusters=20))])
X = spotify_data.select_dtypes(np.number)#only selects the numbers

song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
spotify_data["cluster_label"] = song_cluster_labels

#Visualising the pipelines implementation of Songs clusters into  a two dimensional space using PCA

#from sklearn.manifold import TSNE
#tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2,verbose=2))])
#song_embedding = tsne_pipeline.fit_transform(X)
#projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
#projection['title'] = spotify_data['name']
#projection['cluster'] = spotify_data['cluster_label']

#Visualisation using PCA because the dataset is too large couldnt complete using TSNE
from sklearn.decomposition import PCA

#Initialising a new dataset with a two dimensional pipeline.
pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)

#Creating a panda object to implement a 2-dimensional labeled data structure.
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)

#Updating values of generes and clusters from the kmeans implementation rather than from the 2D implementaion.
projection['title'] = spotify_data['name']
projection['cluster'] = spotify_data['cluster_label']

#Plotting the projections of songs using plotly
import plotly.express as px
#Using the fig variable to plot a scatter plot
fig = px.scatter(projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
fig.show()

from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict

auth_manager = SpotifyClientCredentials(client_id=os.environ["SPOTIFY_CLIENT_ID"],
                                        client_secret=os.environ["SPOTIFY_CLIENT_SECRET"])
sp = spotipy.Spotify(auth_manager=auth_manager)

def fetch_song_data(name, year):
    """
    This function returns a dictionary with data for a song given the name and release year.
    The function uses Spotipy to fetch audio features and metadata for the specified song.
    """
    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name, year), limit=1)
    
    if not results['tracks']['items']:
        return None
    
    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]
    
    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]
    
    for key, value in audio_features.items():
        song_data[key] = value
    
    return dict(song_data)

from collections import defaultdict
from scipy.spatial.distance import cdist
import difflib
number_cols = ['valence','year','acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

def get_song_data(song, spotify_data):
    '''gets all the song details for the selected songs.
    Song takes in input as key value pairs dictionary in form of name and year'''
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    
    except IndexError:
        return fetch_song_data(song['name'], song['year']) 



def get_mean_vector(song_list, spotify_data):
    """Gets the mean vector for a list of songs"""
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song,spotify_data)
        if song_data is None:
            print("Warning, Does not exist in spotify database".format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)
        song_matrix = np.array(list(song_vectors))
        return np.mean(song_matrix, axis = 0)

from collections import defaultdict
def flatten_dict_list(dict_list):
    
    """Utility function for flattening a list of dictionaries"""

    flattened_dict = defaultdict(list)
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
    return dict(flattened_dict)


def recommend_songs(song_list, spotify_data, song_cluster_pipeline, n_songs=10):
    """
    Recommends songs based on a list of previous songs that a user has listened to.
    """
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline['scaler']
    scaled_data = scaler.transform(spotify_data[song_cluster_pipeline['number_cols']])


recommend_songs([{'name': 'Come As You Are', 'year':1991},
                {'name': 'Smells Like Teen Spirit', 'year': 1991},
                {'name': 'Lithium', 'year': 1992},
                {'name': 'All Apologies', 'year': 1993},
                {'name': 'Stay Away', 'year': 1993}],  spotify_data)
