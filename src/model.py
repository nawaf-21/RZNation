import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path

class LyricsModel:
    def __init__(self, model_dir='models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.mood_model = None
        self.artist_model = None
        
    def prepare_features(self, df, target_col='mood_category'):
        """Prepare features for model training"""
        # Vectorize lyrics
        X = self.vectorizer.fit_transform(df['cleaned_lyrics'])
        
        # Get target variable
        y = df[target_col]
        
        return X, y
    
    def train_mood_model(self, df, test_size=0.2, random_state=42):
        """Train a model to predict mood categories"""
        # Prepare features
        X, y = self.prepare_features(df, 'mood_category')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        self.mood_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state
        )
        self.mood_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.mood_model.predict(X_test)
        print("\nMood Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        self._save_model('mood_model.joblib', self.mood_model)
        self._save_model('mood_vectorizer.joblib', self.vectorizer)
        
        return self.mood_model
    
    def train_artist_model(self, df, test_size=0.2, random_state=42):
        """Train a model to predict artists"""
        # Prepare features
        X, y = self.prepare_features(df, 'Artist')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        self.artist_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=random_state
        )
        self.artist_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.artist_model.predict(X_test)
        print("\nArtist Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        self._save_model('artist_model.joblib', self.artist_model)
        self._save_model('artist_vectorizer.joblib', self.vectorizer)
        
        return self.artist_model
    
    def predict_mood(self, lyrics):
        """Predict mood for new lyrics"""
        if self.mood_model is None:
            self._load_model('mood_model.joblib', 'mood')
            self._load_model('mood_vectorizer.joblib', 'vectorizer')
            
        # Vectorize lyrics
        X = self.vectorizer.transform([lyrics])
        
        # Predict
        return self.mood_model.predict(X)[0]
    
    def predict_artist(self, lyrics):
        """Predict artist for new lyrics"""
        if self.artist_model is None:
            self._load_model('artist_model.joblib', 'artist')
            self._load_model('artist_vectorizer.joblib', 'vectorizer')
            
        # Vectorize lyrics
        X = self.vectorizer.transform([lyrics])
        
        # Predict
        return self.artist_model.predict(X)[0]
    
    def _save_model(self, filename, model):
        """Save model to disk"""
        joblib.dump(model, self.model_dir / filename)
        
    def _load_model(self, filename, model_type):
        """Load model from disk"""
        model_path = self.model_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model file {filename} not found")
            
        if model_type == 'mood':
            self.mood_model = joblib.load(model_path)
        elif model_type == 'artist':
            self.artist_model = joblib.load(model_path)
        elif model_type == 'vectorizer':
            self.vectorizer = joblib.load(model_path) 