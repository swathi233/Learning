# Music_recommendation_system
# Music Recommender System

This is a Streamlit-based Music Recommender System that uses Spotify and Genius APIs to recommend songs, display album covers, and fetch lyrics.

## Features

- Recommend songs based on a selected song.
- Display album covers of recommended songs.
- Fetch and display lyrics of recommended songs.
- Provide Spotify URIs for recommended songs.

## Installation
 
### Prerequisites

- Python 3.7+
- A Spotify Developer account
- A Genius Developer account

### Clone the Repository

```sh
git clone https://github.com/your-username/music-recommender-system.git
cd music-recommender-system

### Install Dependencies
! pip install -r requirements.txt


Make sure to replace placeholders like `YOUR_SPOTIFY_CLIENT_ID`, `YOUR_SPOTIFY_CLIENT_SECRET`, and `YOUR_GENIUS_API_TOKEN` with your actual credentials. This setup will provide a clear guide for anyone looking to use or contribute to your project.

Configuration
Spotify API:

1. Create a Spotify Developer account and set up a new application to get your CLIENT_ID and CLIENT_SECRET
2. Replace "YOUR_SPOTIFY_CLIENT_ID" and "YOUR_SPOTIFY_CLIENT_SECRET" in the code with your credentials.

Genius API:

1. Sign up for a Genius API account and obtain an access token.
2. Replace "YOUR_GENIUS_API_TOKEN" in the code with your access token.


## Make sure to generate the pickle files with the ipynb files and then run the app.py

## Run the Streamlit app
streamlit run app.py