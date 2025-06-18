import re
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import streamlit as st

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except:
        pass

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    try:
        nltk.download('wordnet', quiet=True)
    except:
        pass

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    try:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    except:
        pass

class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add common sentiment-neutral words to stop words
        self.stop_words.update([
            'would', 'could', 'should', 'might', 'must', 'shall', 'will',
            'one', 'two', 'first', 'second', 'also', 'said', 'say',
            'get', 'go', 'come', 'see', 'know', 'think', 'take', 'make'
        ])
    
    def clean_text(self, text):
        """
        Clean and preprocess text
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'"()]', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_keywords(self, text, num_keywords=10, min_length=3):
        """
        Extract meaningful keywords from text
        
        Args:
            text (str): Input text
            num_keywords (int): Number of keywords to extract
            min_length (int): Minimum word length
            
        Returns:
            list: List of extracted keywords
        """
        if not text:
            return []
        
        try:
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Try NLTK tokenization first, fallback to simple tokenization
            try:
                tokens = word_tokenize(cleaned_text)
            except:
                # Fallback: simple regex-based tokenization
                tokens = re.findall(r'\b[a-zA-Z]+\b', cleaned_text.lower())
            
            # Remove stopwords and short words
            filtered_tokens = [
                word for word in tokens 
                if word not in self.stop_words 
                and len(word) >= min_length
                and word.isalpha()  # Only alphabetic words
            ]
            
            # Try POS tagging, fallback to frequency-based filtering
            try:
                pos_tags = pos_tag(filtered_tokens)
                # Filter by POS tags (focus on meaningful words)
                meaningful_words = [
                    word for word, pos in pos_tags 
                    if pos.startswith(('NN', 'JJ', 'VB', 'RB'))  # Nouns, adjectives, verbs, adverbs
                ]
            except:
                # Fallback: use all filtered tokens
                meaningful_words = filtered_tokens
            
            # Try lemmatization, fallback to original words
            try:
                lemmatized_words = [self.lemmatizer.lemmatize(word) for word in meaningful_words]
            except:
                lemmatized_words = meaningful_words
            
            # Count frequency and get most common
            word_freq = Counter(lemmatized_words)
            keywords = [word for word, count in word_freq.most_common(num_keywords)]
            
            return keywords
            
        except Exception as e:
            # Final fallback: simple word extraction
            try:
                words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + ',}\b', text.lower())
                filtered_words = [w for w in words if w not in self.stop_words]
                word_freq = Counter(filtered_words)
                return [word for word, count in word_freq.most_common(num_keywords)]
            except:
                return []
    
    def extract_sentiment_words(self, text):
        """
        Extract words that likely contribute to sentiment
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary with positive and negative words
        """
        # Basic sentiment word lists (can be expanded)
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied',
            'perfect', 'awesome', 'brilliant', 'outstanding', 'superb',
            'delighted', 'thrilled', 'excited', 'impressed', 'recommend'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate',
            'dislike', 'disappointed', 'frustrated', 'angry', 'upset',
            'poor', 'worst', 'pathetic', 'useless', 'annoying',
            'boring', 'stupid', 'ridiculous', 'waste', 'avoid'
        }
        
        # Clean and tokenize text
        cleaned_text = self.clean_text(text)
        try:
            tokens = word_tokenize(cleaned_text)
        except:
            # Fallback tokenization
            tokens = re.findall(r'\b[a-zA-Z]+\b', cleaned_text.lower())
        
        # Find sentiment words
        found_positive = [word for word in tokens if word in positive_words]
        found_negative = [word for word in tokens if word in negative_words]
        
        return {
            'positive': found_positive,
            'negative': found_negative
        }
    
    def get_text_statistics(self, text):
        """
        Get basic statistics about the text
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Text statistics
        """
        if not text:
            return {}
        
        try:
            # Basic counts
            char_count = len(text)
            
            # Try NLTK tokenization, fallback to simple methods
            try:
                word_count = len(word_tokenize(text))
                sentence_count = len(sent_tokenize(text))
                words = word_tokenize(text.lower())
            except:
                # Fallback methods
                words = re.findall(r'\b\w+\b', text.lower())
                word_count = len(words)
                sentences = re.split(r'[.!?]+', text)
                sentence_count = len([s for s in sentences if s.strip()])
            
            # Average word length
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            
            # Reading time estimate (average 200 words per minute)
            reading_time = word_count / 200
            
            return {
                'character_count': char_count,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'average_word_length': round(avg_word_length, 2),
                'estimated_reading_time': round(reading_time, 2)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def extract_phrases(self, text, phrase_length=2):
        """
        Extract meaningful phrases from text
        
        Args:
            text (str): Input text
            phrase_length (int): Length of phrases to extract
            
        Returns:
            list: List of phrases
        """
        if not text:
            return []
        
        try:
            # Clean and tokenize
            cleaned_text = self.clean_text(text)
            tokens = word_tokenize(cleaned_text)
            
            # Remove stopwords
            filtered_tokens = [
                word for word in tokens 
                if word not in self.stop_words and word.isalpha()
            ]
            
            # Generate phrases
            phrases = []
            for i in range(len(filtered_tokens) - phrase_length + 1):
                phrase = ' '.join(filtered_tokens[i:i + phrase_length])
                phrases.append(phrase)
            
            # Count frequency and return most common
            phrase_freq = Counter(phrases)
            common_phrases = [phrase for phrase, count in phrase_freq.most_common(10)]
            
            return common_phrases
            
        except Exception as e:
            return []
    
    def highlight_sentiment_words(self, text):
        """
        Highlight potential sentiment-bearing words in text
        
        Args:
            text (str): Input text
            
        Returns:
            str: HTML-formatted text with highlighted words
        """
        sentiment_words = self.extract_sentiment_words(text)
        
        highlighted_text = text
        
        # Highlight positive words
        for word in sentiment_words['positive']:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted_text = pattern.sub(
                f'<span style="background-color: #c8e6c9; padding: 1px 3px; border-radius: 2px;">{word}</span>',
                highlighted_text
            )
        
        # Highlight negative words
        for word in sentiment_words['negative']:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted_text = pattern.sub(
                f'<span style="background-color: #ffcdd2; padding: 1px 3px; border-radius: 2px;">{word}</span>',
                highlighted_text
            )
        
        return highlighted_text
    
    def detect_language(self, text):
        """
        Simple language detection (basic implementation)
        
        Args:
            text (str): Input text
            
        Returns:
            str: Detected language (simplified)
        """
        # This is a very basic implementation
        # In production, you'd use a proper language detection library
        
        if not text:
            return "unknown"
        
        try:
            # Count English words (very basic approach)
            english_words = set(stopwords.words('english'))
            try:
                words = word_tokenize(text.lower())
            except:
                words = re.findall(r'\b\w+\b', text.lower())
                
            english_word_count = sum(1 for word in words if word in english_words)
            
            if len(words) > 0 and english_word_count / len(words) > 0.3:
                return "english"
            else:
                return "non-english"
        except:
            return "unknown"
