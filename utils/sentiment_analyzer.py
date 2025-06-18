import os
import streamlit as st
import re
from collections import Counter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.model_name = "Rule-based Sentiment Analyzer"
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentiment analysis model"""
        try:
            # Load sentiment lexicons
            self.positive_words = {
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'perfect',
                'awesome', 'brilliant', 'outstanding', 'superb', 'delighted',
                'thrilled', 'excited', 'impressed', 'recommend', 'best', 'beautiful',
                'nice', 'incredible', 'marvelous', 'magnificent', 'stellar',
                'phenomenal', 'exceptional', 'remarkable', 'splendid', 'terrific',
                'fabulous', 'divine', 'lovely', 'charming', 'pleasant', 'fun',
                'cool', 'sweet', 'fresh', 'clean', 'smooth', 'fast', 'easy',
                'comfortable', 'cozy', 'warm', 'friendly', 'helpful', 'kind',
                'generous', 'honest', 'reliable', 'trustworthy', 'professional'
            }
            
            self.negative_words = {
                'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate',
                'dislike', 'disappointed', 'frustrated', 'angry', 'upset', 'poor',
                'worst', 'pathetic', 'useless', 'annoying', 'boring', 'stupid',
                'ridiculous', 'waste', 'avoid', 'nasty', 'ugly', 'dirty', 'slow',
                'expensive', 'cheap', 'broken', 'damaged', 'defective', 'faulty',
                'uncomfortable', 'difficult', 'hard', 'confusing', 'complicated',
                'rude', 'mean', 'harsh', 'cruel', 'unfair', 'dishonest',
                'unreliable', 'unprofessional', 'sucks', 'fails', 'disappointing'
            }
            
            # Intensity modifiers
            self.intensifiers = {
                'very': 1.5, 'extremely': 2.0, 'absolutely': 1.8, 'completely': 1.7,
                'totally': 1.6, 'really': 1.4, 'quite': 1.2, 'rather': 1.1,
                'pretty': 1.1, 'incredibly': 1.9, 'amazingly': 1.8, 'exceptionally': 1.9
            }
            
            self.diminishers = {
                'slightly': 0.7, 'somewhat': 0.8, 'kind of': 0.8, 'sort of': 0.8,
                'a bit': 0.8, 'a little': 0.7, 'barely': 0.5, 'hardly': 0.4
            }
            
            logger.info(f"Model {self.model_name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise Exception("Unable to initialize sentiment analysis model")
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of given text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis result with label and score
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Clean and preprocess text
        text = self._preprocess_text(text)
        
        try:
            # Calculate sentiment score
            sentiment_score = self._calculate_sentiment_score(text)
            
            # Determine label and confidence
            if sentiment_score > 0.1:
                label = 'POSITIVE'
                confidence = min(0.95, 0.6 + abs(sentiment_score) * 0.35)
            elif sentiment_score < -0.1:
                label = 'NEGATIVE'
                confidence = min(0.95, 0.6 + abs(sentiment_score) * 0.35)
            else:
                label = 'NEUTRAL'
                confidence = max(0.5, 0.8 - abs(sentiment_score) * 0.3)
            
            # Create all_scores for compatibility
            all_scores = [
                {'label': 'POSITIVE', 'score': confidence if label == 'POSITIVE' else 1-confidence},
                {'label': 'NEGATIVE', 'score': confidence if label == 'NEGATIVE' else 1-confidence},
                {'label': 'NEUTRAL', 'score': confidence if label == 'NEUTRAL' else 1-confidence}
            ]
            
            return {
                'label': label,
                'score': confidence,
                'all_scores': all_scores
            }
            
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {str(e)}")
            raise Exception(f"Sentiment analysis failed: {str(e)}")
    
    def _calculate_sentiment_score(self, text):
        """Calculate sentiment score using rule-based approach"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        positive_score = 0
        negative_score = 0
        
        for i, word in enumerate(words):
            # Check for intensifiers/diminishers before sentiment words
            modifier = 1.0
            if i > 0:
                prev_word = words[i-1]
                if prev_word in self.intensifiers:
                    modifier = self.intensifiers[prev_word]
                elif prev_word in self.diminishers:
                    modifier = self.diminishers[prev_word]
            
            # Score sentiment words
            if word in self.positive_words:
                positive_score += modifier
            elif word in self.negative_words:
                negative_score += modifier
        
        # Normalize scores
        total_words = len(words)
        if total_words > 0:
            positive_score = positive_score / total_words
            negative_score = negative_score / total_words
        
        # Calculate final sentiment score
        sentiment_score = positive_score - negative_score
        
        return sentiment_score
    
    def _preprocess_text(self, text):
        """
        Preprocess text for better analysis
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Preprocessed text
        """
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Handle encoding issues
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        return text
    
    def batch_analyze(self, texts, batch_size=8):
        """
        Analyze multiple texts in batches
        
        Args:
            texts (list): List of texts to analyze
            batch_size (int): Number of texts to process at once
            
        Returns:
            list: List of sentiment analysis results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Process batch
                batch_results = []
                for text in batch:
                    result = self.analyze_sentiment(text)
                    batch_results.append(result)
                
                results.extend(batch_results)
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                # Add error placeholders for failed batch
                for _ in batch:
                    results.append({
                        'label': 'ERROR',
                        'score': 0.0,
                        'error': str(e)
                    })
        
        return results
    
    def get_model_info(self):
        """
        Get information about the loaded model
        
        Returns:
            dict: Model information
        """
        return {
            'model_name': self.model_name,
            'device': 'CPU',
            'max_length': 5000,
            'labels': ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        }
    
    def validate_text_length(self, text):
        """
        Validate if text is within acceptable length
        
        Args:
            text (str): Text to validate
            
        Returns:
            tuple: (is_valid, message)
        """
        max_length = 5000
        
        if len(text) > max_length:
            return False, f"Text too long ({len(text)} characters, max {max_length})"
        
        return True, "Text length acceptable"
