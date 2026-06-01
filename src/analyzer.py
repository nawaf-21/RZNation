import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class LyricsAnalyzer:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def get_word_frequencies(self, df, column='processed_tokens', top_n=20):
        """Get most common words for each artist"""
        word_freq = {}
        for artist in df['Artist'].unique():
            artist_tokens = df[df['Artist'] == artist][column].explode()
            word_freq[artist] = Counter(artist_tokens).most_common(top_n)
        return word_freq
    
    def perform_topic_modeling(self, df, column='cleaned_lyrics', n_topics=5):
        """Perform LDA topic modeling on lyrics"""
        # Create TF-IDF matrix
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(df[column])
        
        # Perform LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42
        )
        lda_output = lda.fit_transform(tfidf_matrix)
        
        # Get feature names
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Get top words for each topic
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10-1:-1]]
            topics.append(top_words)
            
        return topics, lda_output
    
    def calculate_artist_similarity(self, df, column='cleaned_lyrics'):
        """Calculate similarity between artists based on their lyrics"""
        # Create TF-IDF matrix for each artist's combined lyrics
        artist_lyrics = df.groupby('Artist')[column].apply(' '.join)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(artist_lyrics)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Create similarity network
        artists = artist_lyrics.index
        similarity_network = nx.Graph()
        
        # Add nodes
        for artist in artists:
            similarity_network.add_node(artist)
            
        # Add edges (only top 3 most similar artists for each artist)
        for i, artist1 in enumerate(artists):
            similarities = similarity_matrix[i]
            top_indices = np.argsort(similarities)[-4:-1]  # -1 to exclude self
            for idx in top_indices:
                artist2 = artists[idx]
                similarity_network.add_edge(
                    artist1, 
                    artist2, 
                    weight=similarities[idx]
                )
                
        return similarity_network
    
    def analyze_temporal_trends(self, df):
        """Analyze how lyrics characteristics change over time"""
        years = pd.to_numeric(df['Year'], errors='coerce')
        if years.isna().all():
            raise ValueError("No valid year values found in the Year column")

        # Group by year and calculate statistics
        yearly_stats = df.groupby(years.astype('Int64')).agg({
            'sentiment': ['mean', 'std'],
            'subjectivity': ['mean', 'std'],
            'lyrics_length': ['mean', 'std'],
            'word_count': ['mean', 'std']
        })
        
        return yearly_stats
    
    def get_artist_statistics(self, df):
        """Get comprehensive statistics for each artist"""
        artist_stats = df.groupby('Artist').agg({
            'sentiment': ['mean', 'std', 'min', 'max'],
            'subjectivity': ['mean', 'std'],
            'lyrics_length': ['mean', 'std'],
            'word_count': ['mean', 'std'],
            'Title': 'count'  # number of songs
        })
        
        # Calculate mood distribution
        mood_dist = df.groupby(['Artist', 'mood_category']).size().unstack(fill_value=0)
        mood_dist = mood_dist.div(mood_dist.sum(axis=1), axis=0)  # normalize
        
        return artist_stats, mood_dist 