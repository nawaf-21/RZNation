import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

def _ensure_nltk_data():
    """Download NLTK corpora if missing (supports NLTK 3.8+ punkt_tab)."""
    resources = [
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


class LyricsPreprocessor:
    def __init__(self):
        _ensure_nltk_data()
            
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
        
    def preprocess_lyrics(self, df, lyrics_column='Lyric'):
        """Preprocess lyrics and add new features"""
        # Clean lyrics
        df['cleaned_lyrics'] = df[lyrics_column].apply(self.clean_text)
        
        # Tokenize
        df['tokens'] = df['cleaned_lyrics'].apply(word_tokenize)
        
        # Remove stopwords and lemmatize
        df['processed_tokens'] = df['tokens'].apply(
            lambda x: [self.lemmatizer.lemmatize(word) for word in x 
                      if word not in self.stop_words]
        )
        
        # Calculate basic features
        df['lyrics_length'] = df['cleaned_lyrics'].apply(len)
        df['word_count'] = df['tokens'].apply(len)
        df['unique_words'] = df['processed_tokens'].apply(lambda x: len(set(x)))
        df['avg_word_length'] = df['cleaned_lyrics'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x else 0
        )
        
        # Sentiment analysis
        df['sentiment'] = df['cleaned_lyrics'].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )
        df['subjectivity'] = df['cleaned_lyrics'].apply(
            lambda x: TextBlob(x).sentiment.subjectivity
        )
        
        return df
        
    def create_mood_labels(self, df, sentiment_column='sentiment'):
        """Create mood labels based on sentiment scores"""
        df['mood_category'] = pd.cut(
            df[sentiment_column],
            bins=[-1, -0.33, 0.33, 1],
            labels=['Sad', 'Neutral', 'Happy']
        )
        return df 