#This code implements spotipy to implement the data from spotify on python
#And create a user freindly way to give a music reccomendation 

#Installing neccesaary python libraries to implement the program 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spotipy
import os


#Reading the dataset files
spotify_data = pd.read_csv('./data/data.csv.zip')
genre_data = pd.read_csv('./data/data_by_genres.csv')
data_by_year = pd.read_csv('./data/data_by_year.csv')

#We are using plotly to visualise (Using a line graph) the values of different audios.
import plotly.express as px 
sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
fig = px.line(data_by_year, x='year', y=sound_features)
fig.show()