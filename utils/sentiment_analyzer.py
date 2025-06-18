import os
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentiment analysis model"""
        try:
            # Check if CUDA is available
            device = 0 if torch.cuda.is_available() else -1
            
            # Load the model with caching
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=device,
                return_all_scores=True
            )
            
            # Load tokenizer for text length checking
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info(f"Model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Fallback to a simpler model
            try:
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=device,
                    return_all_scores=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
                logger.info("Fallback model loaded successfully")
            except Exception as fallback_error:
                logger.error(f"Fallback model failed: {str(fallback_error)}")
                raise Exception("Unable to load any sentiment analysis model")
    
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
        
        # Check text length
        if len(text) > 5000:  # Rough character limit
            text = text[:5000]  # Truncate if too long
        
        try:
            # Get all sentiment scores
            results = self.pipeline(text)
            
            # Handle different model outputs
            if isinstance(results[0], list):
                # Model returns all scores
                scores = results[0]
            else:
                # Model returns single result
                scores = results
            
            # Find the highest confidence score
            best_result = max(scores, key=lambda x: x['score'])
            
            # Map labels to standard format
            label_mapping = {
                'LABEL_0': 'NEGATIVE',
                'LABEL_1': 'NEUTRAL', 
                'LABEL_2': 'POSITIVE',
                'NEGATIVE': 'NEGATIVE',
                'POSITIVE': 'POSITIVE',
                'NEUTRAL': 'NEUTRAL'
            }
            
            mapped_label = label_mapping.get(best_result['label'], best_result['label'])
            
            return {
                'label': mapped_label,
                'score': best_result['score'],
                'all_scores': scores
            }
            
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {str(e)}")
            raise Exception(f"Sentiment analysis failed: {str(e)}")
    
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
            'device': 'CUDA' if torch.cuda.is_available() else 'CPU',
            'max_length': self.tokenizer.model_max_length if self.tokenizer else 512,
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
        if not self.tokenizer:
            return True, "Length validation not available"
        
        try:
            tokens = self.tokenizer.encode(text)
            max_length = self.tokenizer.model_max_length
            
            if len(tokens) > max_length:
                return False, f"Text too long ({len(tokens)} tokens, max {max_length})"
            
            return True, "Text length acceptable"
            
        except Exception as e:
            return True, f"Length validation failed: {str(e)}"
