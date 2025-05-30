import pickle
import streamlit as st    
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import lyricsgenius

# Spotify credentials
CLIENT_ID = "YOUR_SPOTIFY_CLIENT_ID"
CLIENT_SECRET = "YOUR_SPOTIFY_CLIENT_SECRET"
 
# Initialize Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
 
# Initialize Genius client
GENIUS_API_TOKEN = "YOUR_GENIUS_API_TOKEN"
genius = lyricsgenius.Genius(GENIUS_API_TOKEN)
 
def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        return album_cover_url
    else:
        return "https://i.postimg.cc/0QNxYz4V/social.png"

def get_song_lyrics(song_name, artist_name):
    try:
        song = genius.search_song(song_name, artist_name)
        if song:
            return song.lyrics
        else:
            return "Lyrics not found."
    except Exception as e:
        return f"Error fetching lyrics: {str(e)}"

def recommend(song):
    index = music[music['song'] == song].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_music_names = []
    recommended_music_posters = []
    recommended_music_uris = []
    recommended_music_lyrics = []
    for i in distances[1:6]:
        artist = music.iloc[i[0]].artist
        song_name = music.iloc[i[0]].song
        recommended_music_posters.append(get_song_album_cover_url(song_name, artist))
        recommended_music_names.append(song_name)
        recommended_music_lyrics.append(get_song_lyrics(song_name, artist))
        
        search_query = f"track:{song_name} artist:{artist}"
        results = sp.search(q=search_query, type="track")
        if results and results["tracks"]["items"]:
            track_uri = results["tracks"]["items"][0]["uri"]
            recommended_music_uris.append(track_uri)
        else:
            recommended_music_uris.append(None)

    return recommended_music_names, recommended_music_posters, recommended_music_uris, recommended_music_lyrics

st.header('Music Recommender System')
music = pickle.load(open('df.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

music_list = music['song'].values
selected_song = st.selectbox(
    "Type or select a song from the dropdown",
    music_list
)

if st.button('Show Recommendation'):
    recommended_music_names, recommended_music_posters, recommended_music_uris, recommended_music_lyrics = recommend(selected_song)
    for i in range(5):
        st.write(f"**{recommended_music_names[i]}**")
        st.image(recommended_music_posters[i])
        if recommended_music_uris[i]:
            st.components.v1.iframe(f"https://open.spotify.com/embed/track/{recommended_music_uris[i].split(':')[-1]}", height=80)
        st.write(f"**Lyrics:**\n{recommended_music_lyrics[i]}")
