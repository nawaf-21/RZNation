import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import os
from collections import Counter
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from src.preprocessor import _ensure_nltk_data

_ensure_nltk_data()

def load_all_lyrics():
    """Load all lyrics from CSV files in the csv directory"""
    lyrics_data = []
    csv_dir = 'csv'
    
    for file in os.listdir(csv_dir):
        if file.endswith('.csv'):
            artist_name = file.replace('.csv', '')
            df = pd.read_csv(os.path.join(csv_dir, file))
            df['Artist'] = artist_name
            lyrics_data.append(df)
    
    return pd.concat(lyrics_data, ignore_index=True)

def create_sentiment_distribution():
    """Create sentiment distribution plot for each artist"""
    df = load_all_lyrics()
    
    # Calculate sentiment for each song
    df['Sentiment'] = df['Lyric'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    # Create violin plot
    fig = px.violin(df, y='Sentiment', x='Artist', box=True, points="all",
                   title='Sentiment Distribution Across Artists',
                   labels={'Sentiment': 'Sentiment Score (-1 to 1)', 'Artist': 'Artist Name'},
                   color='Artist')
    
    fig.update_layout(
        showlegend=False,
        title_x=0.5,
        height=800
    )
    
    fig.write_html("plots/sentiment_distribution.html")

def create_word_count_analysis():
    """Create word count analysis visualization"""
    df = load_all_lyrics()
    
    # Calculate average word count per song for each artist
    df['Word_Count'] = df['Lyric'].apply(lambda x: len(str(x).split()))
    avg_words = df.groupby('Artist')['Word_Count'].mean().sort_values(ascending=True)
    
    fig = px.bar(avg_words,
                 title='Average Words per Song by Artist',
                 labels={'value': 'Average Word Count', 'Artist': 'Artist Name'},
                 orientation='h')
    
    fig.update_layout(
        showlegend=False,
        title_x=0.5,
        height=800,
        xaxis_title="Average Word Count",
        yaxis_title="Artist"
    )
    
    fig.write_html("plots/word_count_analysis.html")

def create_common_words_bubble():
    """Create bubble chart of most common words"""
    df = load_all_lyrics()
    stop_words = set(stopwords.words('english'))
    
    # Combine all lyrics and tokenize
    all_words = ' '.join(df['Lyric'].astype(str)).lower()
    tokens = word_tokenize(all_words)
    
    # Remove stop words and count words
    filtered_words = [word for word in tokens if word.isalnum() and word not in stop_words]
    word_freq = Counter(filtered_words).most_common(50)
    
    # Create bubble chart
    fig = px.scatter(
        x=[item[1] for item in word_freq],
        y=[item[0] for item in word_freq],
        size=[item[1] for item in word_freq],
        text=[item[0] for item in word_freq],
        title='50 Most Common Words in Lyrics',
        labels={'x': 'Frequency', 'y': 'Word'}
    )
    
    fig.update_traces(textposition='middle center')
    fig.update_layout(
        title_x=0.5,
        height=800,
        showlegend=False
    )
    
    fig.write_html("plots/common_words_bubble.html")

def create_artist_song_count():
    """Create pie chart of song count by artist"""
    df = load_all_lyrics()
    song_counts = df['Artist'].value_counts()
    
    fig = px.pie(values=song_counts.values,
                 names=song_counts.index,
                 title='Distribution of Songs by Artist')
    
    fig.update_layout(
        title_x=0.5,
        height=800
    )
    
    fig.write_html("plots/artist_song_distribution.html")

def create_sentiment_timeline():
    """Create sentiment timeline analysis"""
    df = load_all_lyrics()
    df['Sentiment'] = df['Lyric'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    # Create line plot for sentiment over time
    fig = go.Figure()
    
    for artist in df['Artist'].unique():
        artist_data = df[df['Artist'] == artist]
        fig.add_trace(go.Scatter(
            y=artist_data['Sentiment'].rolling(window=5).mean(),
            name=artist,
            mode='lines',
            hovertext=artist_data['Lyric'].str[:100]  # Show first 100 chars of lyrics on hover
        ))
    
    fig.update_layout(
        title='Sentiment Timeline Analysis',
        title_x=0.5,
        height=800,
        xaxis_title="Song Index",
        yaxis_title="Sentiment Score (Rolling Average)",
        hovermode='x unified'
    )
    
    fig.write_html("plots/sentiment_timeline.html")

def main():
    """Create all visualizations"""
    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    print("Generating visualizations...")
    create_sentiment_distribution()
    create_word_count_analysis()
    create_common_words_bubble()
    create_artist_song_count()
    create_sentiment_timeline()
    print("Visualizations have been generated in the 'plots' directory!")

if __name__ == "__main__":
    main() 