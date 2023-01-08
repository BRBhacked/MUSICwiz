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
tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=2),verbose = True)])
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

song_cluster_pipeline = Pipeline([('scaler',StandardScaler()),('kmeans',KMeans(n_clusters=20,verbose=2))])
X = spotify_data.select_dtypes(np.number)#only selects the numbers

song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
spotify_data["cluster_label"] = song_cluster_labels

#Visualising the pipelines implementation of Songs clusters into  a two dimensional space using PCA

from sklearn.manifold import TSNE
tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=2))])
song_embedding = tsne_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = spotify_data['name']
projection['cluster'] = spotify_data['cluster_label']

#Plotting the projections of songs using plotly

import plotly.express as px
fig = px.scatter(projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
fig.show()
