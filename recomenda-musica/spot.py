import sys
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
import json
import requests


client_id = "x"
client_secret = "x"
redirect_uri = "localhost:/8080/callback"
scope = "playlist-modify-public, playlist-modify-private, user-library-read, user-top-read"
username = "12185589298"
spotifyObject = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,client_secret=client_secret,scope=scope,redirect_uri=redirect_uri))



#Create playlist from user-defined number of top songs

playlist_name = "Teste 1"
playlist_description = "isso é um teste"

spotifyObject.user_playlist_create(user=username,name=playlist_name,public=True,description=playlist_description)

user_input = 10


songliste = []



#Add songs


result = sp.current_user_top_tracks(limit=user_input, offset=100, time_range="medium_term")

for item in result["items"]:
    songliste.append(item["uri"])





prePlaylist = spotifyObject.user_playlists(user=username)
playlist = prePlaylist["items"][0]["id"]



#Create playlist


spotifyObject.playlist_add_items(playlist_id=playlist,items=songliste)