import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import networkx as nx
import numpy as np

class LyricsVisualizer:
    def __init__(self):
        # Set a valid matplotlib style
        plt.style.use('default')  # Use default style instead of seaborn
        sns.set_theme()  # This will set seaborn's default styling
        self.colors = px.colors.qualitative.Set3
        
    def create_wordcloud(self, text, title="Word Cloud"):
        """Create a word cloud from text"""
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            contour_width=3,
            contour_color='steelblue'
        ).generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        return plt.gcf()
    
    def plot_sentiment_distribution(self, df, artist=None):
        """Plot sentiment distribution"""
        if artist:
            data = df[df['Artist'] == artist]
            title = f'Sentiment Distribution for {artist}'
        else:
            data = df
            title = 'Overall Sentiment Distribution'
            
        fig = px.histogram(
            data,
            x='sentiment',
            color='mood_category',
            title=title,
            labels={'sentiment': 'Sentiment Score', 'count': 'Number of Songs'},
            color_discrete_sequence=self.colors
        )
        return fig
    
    def plot_artist_similarity_network(self, similarity_network):
        """Plot artist similarity network"""
        plt.figure(figsize=(12, 8))
        
        # Use spring layout for node positions
        pos = nx.spring_layout(similarity_network)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            similarity_network,
            pos,
            node_size=1000,
            node_color='lightblue',
            alpha=0.7
        )
        
        # Draw edges with varying thickness based on similarity
        edge_weights = [similarity_network[u][v]['weight'] * 2 
                       for u, v in similarity_network.edges()]
        nx.draw_networkx_edges(
            similarity_network,
            pos,
            width=edge_weights,
            alpha=0.4
        )
        
        # Add labels
        nx.draw_networkx_labels(
            similarity_network,
            pos,
            font_size=8,
            font_weight='bold'
        )
        
        plt.title('Artist Similarity Network')
        plt.axis('off')
        return plt.gcf()
    
    def plot_temporal_trends(self, yearly_stats):
        """Plot temporal trends in lyrics characteristics"""
        fig = go.Figure()
        
        # Add sentiment trend
        fig.add_trace(go.Scatter(
            x=yearly_stats.index,
            y=yearly_stats[('sentiment', 'mean')],
            name='Average Sentiment',
            line=dict(color=self.colors[0])
        ))
        
        # Add subjectivity trend
        fig.add_trace(go.Scatter(
            x=yearly_stats.index,
            y=yearly_stats[('subjectivity', 'mean')],
            name='Average Subjectivity',
            line=dict(color=self.colors[1])
        ))
        
        fig.update_layout(
            title='Temporal Trends in Lyrics Characteristics',
            xaxis_title='Year',
            yaxis_title='Score',
            hovermode='x unified'
        )
        
        return fig
    
    def plot_artist_statistics(self, artist_stats, mood_dist):
        """Create dashboard of artist statistics"""
        # Create subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Average sentiment by artist
        plt.subplot(2, 2, 1)
        sns.barplot(
            x=artist_stats.index,
            y=artist_stats[('sentiment', 'mean')],
            palette=self.colors
        )
        plt.xticks(rotation=45)
        plt.title('Average Sentiment by Artist')
        
        # Plot 2: Mood distribution
        plt.subplot(2, 2, 2)
        mood_dist.plot(
            kind='bar',
            stacked=True,
            colormap='Set3',
            ax=plt.gca()
        )
        plt.xticks(rotation=45)
        plt.title('Mood Distribution by Artist')
        plt.legend(title='Mood')
        
        # Plot 3: Average word count
        plt.subplot(2, 2, 3)
        sns.barplot(
            x=artist_stats.index,
            y=artist_stats[('word_count', 'mean')],
            palette=self.colors
        )
        plt.xticks(rotation=45)
        plt.title('Average Word Count by Artist')
        
        # Plot 4: Number of songs
        plt.subplot(2, 2, 4)
        sns.barplot(
            x=artist_stats.index,
            y=artist_stats[('Title', 'count')],
            palette=self.colors
        )
        plt.xticks(rotation=45)
        plt.title('Number of Songs by Artist')
        
        plt.tight_layout()
        return fig 