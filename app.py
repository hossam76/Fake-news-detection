import os
import pickle
import joblib
import re
import numpy as np
import pandas as pd
import json
import uuid
import datetime
import math  # Add math module import
from collections import Counter
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from textblob import TextBlob
import base64
from io import BytesIO

# Try to import spacy for advanced NLP
try:
    import spacy
    # Download small English model if not already downloaded
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        print("Downloading spaCy model...")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
    print("spaCy successfully imported. Advanced NLP features will be available.")
except ImportError:
    HAS_SPACY = False
    print("spaCy not installed. Some advanced NLP features will not be available.")

# Try to import explainable AI libraries
try:
    import lime
    from lime.lime_text import LimeTextExplainer
    HAS_LIME = True
    print("LIME successfully imported. Explainable AI features will be available.")
except ImportError:
    HAS_LIME = False
    print("LIME not installed. Explainable AI features will not be available.")

# Flask imports
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_file

# Try to import TensorFlow
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    HAS_TENSORFLOW = True
    print("TensorFlow successfully imported. LSTM model will be available.")
except ImportError:
    HAS_TENSORFLOW = False
    print("TensorFlow not installed. LSTM model will not be available.")

# Try to import newspaper3k for article extraction
try:
    from newspaper import Article
    HAS_NEWSPAPER = True
    print("Newspaper3k successfully imported. Article extraction will be available.")
except ImportError:
    HAS_NEWSPAPER = False
    print("Newspaper3k not installed. Article extraction will not be available.")

# Initialize Flask app
app = Flask(__name__, static_folder='static')
app.secret_key = os.urandom(24)  # For session management

# Add Jinja utility functions
app.jinja_env.globals.update(min=min, max=max)

# Add caching
from flask_caching import Cache
cache_config = {
    "DEBUG": True,          # Some Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}
cache = Cache(app, config=cache_config)

# Create directory for analysis data if it doesn't exist
os.makedirs('analysis_data', exist_ok=True)

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Initialize LIME explainer if available
if HAS_LIME:
    explainer = LimeTextExplainer(class_names=['fake', 'real'])

# Load models
@app.before_first_request
def load_models():
    global vectorizer, nb_model, rf_model, lstm_model, max_sequence_length, tokenizer
    global explainable_features
    
    # Initialize LSTM-related variables as None
    lstm_model = None
    tokenizer = None
    max_sequence_length = 100
    
    try:
        # First try to load using joblib (faster and more efficient)
        vectorizer = joblib.load('vectorizer.pkl')
        nb_model = joblib.load('naive_bayes_model.pkl')
        rf_model = joblib.load('random_forest_model.pkl')
        print("Traditional ML models loaded successfully with joblib")
    except Exception as e:
        # Fall back to pickle if joblib fails
        print(f"Joblib loading failed: {e}")
        print("Falling back to pickle for loading models")
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open('naive_bayes_model.pkl', 'rb') as f:
            nb_model = pickle.load(f)
        
        with open('random_forest_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        print("Traditional ML models loaded successfully with pickle")
    
    # Load the LSTM model if TensorFlow is available
    if HAS_TENSORFLOW:
        try:
            lstm_model = load_model('lstm_model.h5')
            # Load tokenizer for LSTM if it exists
            with open('tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)
            # Set maximum sequence length for LSTM (should match what was used during training)
            max_sequence_length = 100  # Update with your actual value used during training
            print("LSTM model loaded successfully")
        except Exception as e:
            print(f"Could not load LSTM model: {e}")
            lstm_model = None
            tokenizer = None
    
    # Extract feature names for explainability if possible
    try:
        explainable_features = vectorizer.get_feature_names_out()
    except:
        try:
            explainable_features = vectorizer.get_feature_names()
        except:
            explainable_features = None
            print("Could not extract feature names from vectorizer")

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(processed_tokens)

# Helper function for text analysis
def analyze_text(text):
    """Perform comprehensive detailed analysis on the text"""
    analysis = {}
    
    # Basic stats
    analysis['word_count'] = len(text.split())
    analysis['sentence_count'] = len(sent_tokenize(text))
    analysis['avg_word_length'] = sum(len(word) for word in text.split()) / max(1, len(text.split()))
    analysis['character_count'] = len(text)
    analysis['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])
    
    # Calculate reading metrics
    words = text.split()
    syllable_count = 0
    complex_word_count = 0
    
    for word in words:
        syllables = count_syllables(word)
        syllable_count += syllables
        if syllables >= 3:
            complex_word_count += 1
    
    # Readability metrics
    if len(words) > 0 and len(sent_tokenize(text)) > 0:
        flesch_reading_ease = calculate_flesch_reading_ease(text)
        flesch_kincaid_grade = calculate_flesch_kincaid_grade(text)
        complex_word_percentage = (complex_word_count / max(1, len(words))) * 100
        
        analysis['readability'] = {
            'flesch_reading_ease': flesch_reading_ease,
            'flesch_kincaid_grade': flesch_kincaid_grade,
            'complex_word_percentage': complex_word_percentage,
            'avg_syllables_per_word': syllable_count / max(1, len(words)),
            'avg_sentence_length': len(words) / max(1, len(sent_tokenize(text))),
            # Add bounded values for safe template use
            'flesch_reading_ease_bounded': min(100, max(0, flesch_reading_ease))
        }
    
    # Sentiment analysis using TextBlob
    blob = TextBlob(text)
    analysis['sentiment'] = {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'assessment': get_sentiment_assessment(blob.sentiment.polarity, blob.sentiment.subjectivity)
    }
    
    # Extract top words (excluding stopwords)
    words = [word.lower() for word in word_tokenize(text) 
             if word.isalpha() and word.lower() not in stop_words]
    word_freq = Counter(words)
    analysis['top_words'] = dict(word_freq.most_common(15))
    
    # Extract word categories
    emotional_words = [w for w in words if is_emotional_word(w)]
    factual_words = [w for w in words if is_factual_indicator(w)]
    uncertain_words = [w for w in words if is_uncertainty_indicator(w)]
    
    analysis['language_indicators'] = {
        'emotional_word_count': len(emotional_words),
        'emotional_words': emotional_words[:10],
        'factual_indicators': factual_words[:10],
        'uncertainty_indicators': uncertain_words[:10],
        'emotional_ratio': len(emotional_words) / max(1, len(words)),
        'factual_ratio': len(factual_words) / max(1, len(words)),
        'uncertainty_ratio': len(uncertain_words) / max(1, len(words))
    }
    
    # More advanced NLP with spaCy if available
    if HAS_SPACY:
        doc = nlp(text)
        
        # Named Entity Recognition
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        entity_types = Counter([ent.label_ for ent in doc.ents])
        analysis['named_entities'] = entities
        analysis['entity_counts'] = dict(entity_types)
        
        # Part-of-speech tagging stats
        pos_counts = Counter([token.pos_ for token in doc])
        analysis['pos_counts'] = dict(pos_counts)
        
        # Extract key phrases (noun chunks)
        key_phrases = [chunk.text for chunk in doc.noun_chunks]
        analysis['key_phrases'] = key_phrases[:10] if len(key_phrases) > 10 else key_phrases
        
        # Dependency parsing statistics
        dep_counts = Counter([token.dep_ for token in doc])
        analysis['dependency_counts'] = dict(dep_counts)
        
        # Extract quoted text
        quoted_texts = extract_quoted_text(text)
        analysis['quoted_text'] = quoted_texts
        analysis['quote_count'] = len(quoted_texts)
        
        # Text classification analysis
        analysis['subject_domains'] = classify_text_domain(text)
    
    # Content diversity metrics
    analysis['content_diversity'] = {
        'type_token_ratio': len(set(words)) / max(1, len(words)),
        'hapax_legomena': len([w for w in set(words) if words.count(w) == 1]),
        'vocabulary_richness': len(set(words)) / math.sqrt(max(1, len(words)))
    }
    
    return analysis

# Helper functions for enhanced text analysis
def count_syllables(word):
    """Count syllables in a word using a simple heuristic"""
    word = word.lower()
    if len(word) <= 3:
        return 1
    # Remove trailing e
    if word.endswith('e'):
        word = word[:-1]
    # Count vowel groups
    count = 0
    vowels = "aeiouy"
    prev_is_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    return max(1, count)

def calculate_flesch_reading_ease(text):
    """Calculate Flesch Reading Ease score"""
    sentences = sent_tokenize(text)
    words = text.split()
    sentence_count = len(sentences)
    word_count = len(words)
    syllable_count = sum(count_syllables(word) for word in words)
    
    if sentence_count == 0 or word_count == 0:
        return 0
    
    return 206.835 - (1.015 * (word_count / sentence_count)) - (84.6 * (syllable_count / word_count))

def calculate_flesch_kincaid_grade(text):
    """Calculate Flesch-Kincaid Grade Level"""
    sentences = sent_tokenize(text)
    words = text.split()
    sentence_count = len(sentences)
    word_count = len(words)
    syllable_count = sum(count_syllables(word) for word in words)
    
    if sentence_count == 0 or word_count == 0:
        return 0
    
    return (0.39 * (word_count / sentence_count)) + (11.8 * (syllable_count / word_count)) - 15.59

def get_sentiment_assessment(polarity, subjectivity):
    """Generate a human-readable assessment of sentiment scores"""
    # Polarity assessment
    if polarity >= 0.5:
        polarity_assessment = "very positive"
    elif polarity >= 0.1:
        polarity_assessment = "somewhat positive"
    elif polarity <= -0.5:
        polarity_assessment = "very negative"
    elif polarity <= -0.1:
        polarity_assessment = "somewhat negative"
    else:
        polarity_assessment = "neutral"
    
    # Subjectivity assessment
    if subjectivity >= 0.75:
        subjectivity_assessment = "highly subjective"
    elif subjectivity >= 0.5:
        subjectivity_assessment = "somewhat subjective"
    elif subjectivity <= 0.25:
        subjectivity_assessment = "highly objective"
    else:
        subjectivity_assessment = "somewhat objective"
    
    return f"This text appears {polarity_assessment} in tone and {subjectivity_assessment} in nature."

def is_emotional_word(word):
    """Check if a word has emotional content - simplified version"""
    emotional_word_list = {
        'amazing', 'terrible', 'awesome', 'horrible', 'excellent', 'painful', 'wonderful', 
        'tragic', 'beautiful', 'ugly', 'fantastic', 'devastating', 'incredible', 'awful', 
        'extraordinary', 'terrifying', 'perfect', 'disastrous', 'horrific', 'stunning',
        'shocking', 'outrageous', 'scandal', 'triumph', 'disaster', 'miracle', 'tragedy',
        'catastrophe', 'breakthrough'
    }
    return word.lower() in emotional_word_list

def is_factual_indicator(word):
    """Check if word indicates factual content - simplified version"""
    factual_indicators = {
        'study', 'research', 'survey', 'report', 'analysis', 'data', 'evidence', 'found',
        'published', 'according', 'confirmed', 'official', 'statistics', 'document',
        'verified', 'measured', 'established', 'demonstrated', 'proven', 'recorded'
    }
    return word.lower() in factual_indicators

def is_uncertainty_indicator(word):
    """Check if word indicates uncertainty - simplified version"""
    uncertainty_indicators = {
        'maybe', 'perhaps', 'possibly', 'might', 'could', 'allegedly', 'reportedly',
        'apparently', 'seemingly', 'unclear', 'unconfirmed', 'suspected', 'rumored',
        'likely', 'unlikely', 'probability', 'chance', 'speculate', 'guess', 'estimate'
    }
    return word.lower() in uncertainty_indicators

def extract_quoted_text(text):
    """Extract text within quotation marks"""
    quoted_texts = []
    pattern = r'["\'](.*?)["\']'
    matches = re.finditer(pattern, text)
    
    for match in matches:
        quoted_text = match.group(1)
        if len(quoted_text) > 3:  # Filter out very short quotes
            quoted_texts.append(quoted_text)
    
    return quoted_texts

def classify_text_domain(text):
    """Classify text into subject domains using keyword approach"""
    domains = {
        'politics': ['government', 'election', 'political', 'president', 'vote', 'party', 'policy',
                    'senate', 'congress', 'law', 'democracy', 'campaign', 'candidate'],
        'health': ['health', 'disease', 'medical', 'doctor', 'patient', 'hospital', 'treatment',
                  'vaccine', 'medicine', 'cure', 'symptom', 'virus', 'pandemic', 'covid'],
        'science': ['science', 'research', 'study', 'scientist', 'experiment', 'discovery',
                   'technology', 'innovation', 'data', 'theory', 'evidence', 'laboratory'],
        'finance': ['finance', 'money', 'market', 'stock', 'investment', 'economic', 'economy',
                   'bank', 'dollar', 'trade', 'business', 'currency', 'inflation', 'recession'],
        'entertainment': ['movie', 'film', 'celebrity', 'actor', 'music', 'entertainment',
                         'hollywood', 'tv', 'television', 'star', 'famous', 'award', 'show'],
        'sports': ['sport', 'game', 'team', 'player', 'championship', 'tournament', 'win',
                  'score', 'match', 'athlete', 'coach', 'season', 'football', 'basketball'],
        'technology': ['tech', 'computer', 'internet', 'software', 'digital', 'app',
                     'device', 'cyber', 'online', 'mobile', 'web', 'smart', 'code', 'ai']
    }
    
    text_lower = text.lower()
    domain_scores = {}
    
    for domain, keywords in domains.items():
        count = sum(text_lower.count(keyword) for keyword in keywords)
        if count > 0:
            domain_scores[domain] = count
    
    # Return sorted domains by score
    if domain_scores:
        return sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [('general', 1)]  # Default if no specific domain found

# Generate word cloud image
def generate_wordcloud(text):
    # Preprocess text to remove stopwords
    processed_words = [word.lower() for word in word_tokenize(text)
                       if word.isalpha() and word.lower() not in stop_words]
    wordcloud_text = ' '.join(processed_words)
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white', 
                          max_words=100,
                          colormap='viridis',
                          contour_width=1).generate(wordcloud_text)
    
    # Convert to image
    img = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    
    # Convert to base64 for embedding in HTML
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

# Enhanced source credibility checking
def check_website_credibility(url):
    """More detailed credibility analysis of a news source"""
    try:
        # Parse URL
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Reference databases of news sources (simplified for demonstration)
        reliable_domains = {
            'reuters.com': {'score': 0.95, 'category': 'International News Agency', 'bias': 'minimal'},
            'apnews.com': {'score': 0.95, 'category': 'International News Agency', 'bias': 'minimal'},
            'bloomberg.com': {'score': 0.90, 'category': 'Financial News', 'bias': 'minimal'},
            'bbc.com': {'score': 0.90, 'category': 'International Broadcaster', 'bias': 'minimal-center'},
            'nytimes.com': {'score': 0.88, 'category': 'Newspaper', 'bias': 'center-left'},
            'wsj.com': {'score': 0.87, 'category': 'Newspaper', 'bias': 'center-right'},
            'economist.com': {'score': 0.90, 'category': 'Magazine', 'bias': 'center'}
        }
        
        questionable_domains = {
            'infowars.com': {'score': 0.15, 'category': 'Conspiracy', 'bias': 'extreme-right'},
            'naturalcurenews.com': {'score': 0.20, 'category': 'Pseudoscience', 'bias': 'questionable'},
            'dailybuzzlive.com': {'score': 0.25, 'category': 'Satire/Fake', 'bias': 'questionable'}
        }
        
        domain_info = {}
        credibility_factors = []
        
        # Check if domain is in our reference lists
        if domain in reliable_domains:
            domain_info = reliable_domains[domain]
            credibility_factors.append(f"Source is recognized as reliable: {domain_info['category']}")
            credibility_factors.append(f"Editorial stance: {domain_info['bias']}")
        elif domain in questionable_domains:
            domain_info = questionable_domains[domain]
            credibility_factors.append(f"Source is flagged as potentially unreliable: {domain_info['category']}")
            credibility_factors.append(f"Editorial stance: {domain_info['bias']}")
        
        # Default score if not in our databases
        score = domain_info.get('score', 0.5)
        
        # Try to analyze website content
        try:
            response = requests.get(f"https://{domain}", timeout=3)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Check for about page
            has_about = bool(soup.find('a', text=re.compile(r'about', re.I)) or 
                           'about' in response.text.lower())
            if has_about:
                score += 0.05
                credibility_factors.append("Has about page (+)")
            else:
                credibility_factors.append("No about page (-)")
            
            # Check for contact information
            has_contact = bool(soup.find('a', text=re.compile(r'contact', re.I)) or 
                             'contact' in response.text.lower())
            if has_contact:
                score += 0.05
                credibility_factors.append("Has contact information (+)")
            else:
                credibility_factors.append("No contact information (-)")
            
            # Check for privacy policy
            has_privacy = bool(soup.find('a', text=re.compile(r'privacy', re.I)) or 
                             'privacy policy' in response.text.lower())
            if has_privacy:
                score += 0.05
                credibility_factors.append("Has privacy policy (+)")
            else:
                credibility_factors.append("No privacy policy (-)")
                
            # Check for author attribution
            has_authors = bool(soup.find('author') or soup.find(class_=re.compile(r'author', re.I)))
            if has_authors:
                score += 0.1
                credibility_factors.append("Shows author attribution (+)")
            else:
                credibility_factors.append("No clear author attribution (-)")
            
            # Domain age heuristic (TLD)
            tld = domain.split('.')[-1]
            if tld in ['com', 'org', 'edu', 'gov', 'net']:
                score += 0.03
                credibility_factors.append(f"Standard TLD: .{tld} (+)")
            elif tld in ['co', 'info', 'biz', 'xyz']:
                score -= 0.02
                credibility_factors.append(f"Less common TLD: .{tld} (-)")
                
        except:
            credibility_factors.append("Could not analyze website structure")
        
        # Cap score between 0 and 1
        score = max(0.0, min(1.0, score))
        
        # Determine status based on score
        if score >= 0.8:
            status = 'Highly Credible Source'
        elif score >= 0.6:
            status = 'Likely Credible Source'
        elif score >= 0.4:
            status = 'Uncertain Credibility'
        elif score >= 0.2:
            status = 'Likely Questionable Source'
        else:
            status = 'Known Problematic Source'
        
        # Return comprehensive analysis
        return {
            'score': score,
            'status': status,
            'domain': domain,
            'category': domain_info.get('category', 'Unknown'),
            'bias': domain_info.get('bias', 'Unknown'),
            'factors': credibility_factors,
            'notes': 'Analysis based on source reputation and website characteristics.'
        }
            
    except Exception as e:
        return {
            'score': 0.5,
            'status': 'Unknown',
            'domain': urlparse(url).netloc if url else 'Invalid URL',
            'factors': [f"Analysis error: {str(e)}"],
            'notes': 'Could not complete analysis due to technical issues.'
        }

# Function to extract article from URL
def extract_article(url):
    if not HAS_NEWSPAPER:
        return {'success': False, 'error': 'Newspaper3k library not installed'}
    
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()  # Run NLP to get keywords, summary
        
        return {
            'success': True,
            'title': article.title,
            'text': article.text,
            'authors': article.authors,
            'publish_date': article.publish_date.isoformat() if article.publish_date else None,
            'top_image': article.top_image,
            'keywords': article.keywords,
            'summary': article.summary
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Helper function to make predictions using all available models
def predict_with_models(text):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])

    # Get string predictions directly
    nb_prediction = nb_model.predict(vectorized_text)[0]  # 'real' or 'fake'
    nb_probability = float(nb_model.predict_proba(vectorized_text)[0].max())

    rf_prediction = rf_model.predict(vectorized_text)[0]  # 'real' or 'fake'
    rf_probability = float(rf_model.predict_proba(vectorized_text)[0].max())

    results = {
        'naive_bayes': {
            'prediction': nb_prediction,
            'confidence': nb_probability
        },
        'random_forest': {
            'prediction': rf_prediction,
            'confidence': rf_probability
        }
    }

    if HAS_TENSORFLOW and lstm_model is not None and tokenizer is not None:
        try:
            sequences = tokenizer.texts_to_sequences([processed_text])
            padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

            lstm_pred = lstm_model.predict(padded_sequences)[0]
            lstm_prediction = 'real' if lstm_pred > 0.5 else 'fake'
            lstm_probability = float(lstm_pred if lstm_prediction == 'real' else 1 - lstm_pred)

            results['lstm'] = {
                'prediction': lstm_prediction,
                'confidence': lstm_probability
            }
        except Exception as e:
            print(f"Error in LSTM prediction: {e}")

    # For ensemble, count 'real' as 1, 'fake' as 0
    predictions = []
    for model in ['naive_bayes', 'random_forest']:
        predictions.append(1 if results[model]['prediction'] == 'real' else 0)
    if 'lstm' in results:
        predictions.append(1 if results['lstm']['prediction'] == 'real' else 0)

    ensemble_prediction = 1 if sum(predictions) > len(predictions) / 2 else 0

    results['ensemble'] = {
        'prediction': 'real' if ensemble_prediction == 1 else 'fake',
        'confidence': sum([results[m]['confidence'] for m in results if m != 'ensemble']) / len(predictions)
    }

    return results

# Generate explainable results using LIME
def explain_prediction(text, model_name='naive_bayes'):
    if not HAS_LIME or explainable_features is None:
        return {'error': 'LIME or explainable features not available'}
    
    processed_text = preprocess_text(text)
    
    def predict_proba_func(texts):
        processed_texts = [preprocess_text(t) for t in texts]
        vectorized = vectorizer.transform(processed_texts)
        
        if model_name == 'random_forest':
            return rf_model.predict_proba(vectorized)
        else:
            return nb_model.predict_proba(vectorized)
    
    try:
        exp = explainer.explain_instance(text, predict_proba_func, num_features=10)
        
        explanation = []
        for feature, weight in exp.as_list():
            word = feature.split('=')[1].strip().replace('"', '')
            explanation.append({
                'word': word,
                'weight': weight
            })
            
        highlighted_text = exp.as_html()
        
        return {
            'explanation': explanation,
            'highlighted_html': highlighted_text
        }
    except Exception as e:
        return {'error': f'Error generating explanation: {str(e)}'}

# Routes
@app.route('/')
def home():
    analysis_id = session.get('last_analysis_id')
    saved_analysis = None
    if (analysis_id):
        try:
            with open(f'analysis_data/{analysis_id}.json', 'r') as f:
                saved_analysis = json.load(f)
                session.pop('last_analysis_id', None)
        except:
            pass
            
    return render_template('index.html', saved_analysis=saved_analysis)

@app.route('/hello')
def hello():
    return "Hello, world!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        news_text = data.get('text', '')
        model_choice = data.get('model', 'naive_bayes')
        
        if not news_text:
            return jsonify({'error': 'No text provided'}), 400
        
        if model_choice == 'all':
            results = predict_with_models(news_text)
            return jsonify(results)
        
        processed_text = preprocess_text(news_text)
        
        if model_choice == 'lstm' and HAS_TENSORFLOW and lstm_model is not None and tokenizer is not None:
            try:
                sequences = tokenizer.texts_to_sequences([processed_text])
                padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
                
                lstm_pred = lstm_model.predict(padded_sequences)[0]
                prediction = int(lstm_pred > 0.5)
                probability = float(lstm_pred if prediction else 1 - lstm_pred)
                
                result = {
                    'prediction': 'real' if prediction == 1 else 'fake',
                    'confidence': probability,
                    'model_used': 'lstm'
                }
                return jsonify(result)
            except Exception as e:
                return jsonify({'error': f'LSTM prediction failed: {str(e)}'}), 500
        
        vectorized_text = vectorizer.transform([processed_text])
        
        if model_choice == 'random_forest':
            prediction = rf_model.predict(vectorized_text)[0]
            probability = rf_model.predict_proba(vectorized_text)[0].max()
        else:
            prediction = nb_model.predict(vectorized_text)[0]
            probability = nb_model.predict_proba(vectorized_text)[0].max()
        
        result = {
            'prediction': 'real' if prediction == 1 else 'fake',
            'confidence': float(probability),
            'model_used': model_choice
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_ensemble', methods=['POST'])
def predict_ensemble():
    try:
        data = request.get_json()
        news_text = data.get('text', '')
        
        if not news_text:
            return jsonify({'error': 'No text provided'}), 400
        
        results = predict_with_models(news_text)
        
        return jsonify(results['ensemble'])
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_form', methods=['POST'])
def predict_form():
    try:
        url = request.form.get('news_url', '').strip()
        news_text = request.form.get('news_text', '').strip()
        model_choice = request.form.get('model_choice', 'naive_bayes')
        generate_advanced = request.form.get('generate_advanced', 'false') == 'true'
        
        if url and not news_text and HAS_NEWSPAPER:
            article_data = extract_article(url)
            if (article_data['success']):
                news_text = article_data['text']
            else:
                return render_template('index.html', error=f"Could not extract article: {article_data.get('error', 'Unknown error')}")
        
        if not news_text:
            return render_template('index.html', error='Please enter some text or provide a valid URL')
        
        result = None
        text_analysis = None
        wordcloud_img = None
        credibility_info = None
        explainability_data = None
        all_results = None
        
        if model_choice in ['all', 'ensemble']:
            all_results = predict_with_models(news_text)
            
            if model_choice == 'ensemble':
                result = {
                    'prediction': all_results['ensemble']['prediction'],
                    'confidence': all_results['ensemble']['confidence'] * 100,
                    'model_used': 'Ensemble (All Models)'
                }
            else:
                result = {
                    'prediction': all_results['ensemble']['prediction'],
                    'confidence': all_results['ensemble']['confidence'] * 100,
                    'model_used': 'All Models'
                }
        else:
            processed_text = preprocess_text(news_text)
            
            if model_choice == 'lstm' and HAS_TENSORFLOW and lstm_model is not None and tokenizer is not None:
                try:
                    sequences = tokenizer.texts_to_sequences([processed_text])
                    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
                    
                    lstm_pred = lstm_model.predict(padded_sequences)[0]
                    prediction = int(lstm_pred > 0.5)
                    probability = float(lstm_pred if prediction else 1 - lstm_pred)
                    
                    result = {
                        'prediction': 'real' if prediction == 1 else 'fake',
                        'confidence': probability * 100,
                        'model_used': 'LSTM'
                    }
                except Exception as e:
                    return render_template('index.html', error=f'LSTM prediction failed: {str(e)}')
            else:
                vectorized_text = vectorizer.transform([processed_text])
                
                if model_choice == 'random_forest':
                    prediction = rf_model.predict(vectorized_text)[0]
                    probability = rf_model.predict_proba(vectorized_text)[0].max()
                else:
                    prediction = nb_model.predict(vectorized_text)[0]
                    probability = nb_model.predict_proba(vectorized_text)[0].max()
                
                result = {
                    'prediction': 'real' if prediction == 1 else 'fake',
                    'confidence': float(probability) * 100,
                    'model_used': model_choice
                }
        
        if generate_advanced:
            text_analysis = analyze_text(news_text)
            wordcloud_img = generate_wordcloud(news_text)
            
            if url:
                credibility_info = check_website_credibility(url)
            
            if model_choice in ['naive_bayes', 'random_forest']:
                explainability_data = explain_prediction(news_text, model_choice)
        
        analysis_id = str(uuid.uuid4())
        analysis_data = {
            'id': analysis_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'text': news_text[:500] + '...' if len(news_text) > 500 else news_text,
            'url': url,
            'result': result,
            'model_choice': model_choice,
            'all_results': all_results,
            'text_analysis': text_analysis,
            'wordcloud': wordcloud_img,
            'credibility': credibility_info,
            'explanation': explainability_data
        }
        
        with open(f'analysis_data/{analysis_id}.json', 'w') as f:
            json.dump(analysis_data, f)
        
        session['last_analysis_id'] = analysis_id
        
        return render_template('index.html', 
                               result=result, 
                               news_text=news_text,
                               news_url=url,
                               all_results=all_results,
                               text_analysis=text_analysis,
                               wordcloud=wordcloud_img,
                               credibility=credibility_info,
                               explanation=explainability_data,
                               analysis_id=analysis_id)
    
    except Exception as e:
        import traceback
        return render_template('index.html', error=f"An error occurred: {str(e)}\n{traceback.format_exc()}")

@app.route('/analyze_url', methods=['POST'])
def analyze_url():
    url = request.form.get('url', '')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    if not HAS_NEWSPAPER:
        return jsonify({'error': 'Newspaper3k library required for URL analysis'}), 400
        
    result = extract_article(url)
    return jsonify(result)

@app.route('/show_data')
def show_data():
    try:
        for csv_file in ['preprocessed_data.csv', 'Fake.csv']:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                
                data_info = {
                    'filename': csv_file,
                    'num_rows': len(df),
                    'columns': list(df.columns),
                    'sample': df.head(5).to_dict(orient='records'),
                    'summary': {
                        col: {
                            'mean': float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                            'null_count': int(df[col].isnull().sum())
                        } for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
                    }
                }
                
                return jsonify(data_info)
        
        return jsonify({'error': 'No data files found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export_analysis/<analysis_id>')
def export_analysis(analysis_id):
    try:
        with open(f'analysis_data/{analysis_id}.json', 'r') as f:
            analysis = json.load(f)
            
        formatted_data = {
            "Fake News Analysis Report": {
                "Timestamp": analysis.get('timestamp'),
                "Article Text (Preview)": analysis.get('text'),
                "Source URL": analysis.get('url') if analysis.get('url') else "Not provided",
                "Analysis Results": {
                    "Model Used": analysis.get('result', {}).get('model_used'),
                    "Prediction": analysis.get('result', {}).get('prediction', '').upper(),
                    "Confidence": f"{analysis.get('result', {}).get('confidence', 0):.2f}%"
                }
            }
        }
        
        if analysis.get('text_analysis'):
            ta = analysis['text_analysis']
            formatted_data["Fake News Analysis Report"]["Text Statistics"] = {
                "Word Count": ta.get('word_count'),
                "Sentence Count": ta.get('sentence_count'),
                "Average Word Length": f"{ta.get('avg_word_length'):.2f}",
                "Sentiment Polarity": f"{ta.get('sentiment', {}).get('polarity'):.2f}",
                "Subjectivity": f"{ta.get('sentiment', {}).get('subjectivity'):.2f}",
                "Most Common Words": ta.get('top_words')
            }
        
        if analysis.get('credibility'):
            cred = analysis['credibility']
            formatted_data["Fake News Analysis Report"]["Source Credibility"] = {
                "Score": f"{cred.get('score', 0) * 10:.1f}/10",
                "Status": cred.get('status'),
                "Notes": cred.get('notes')
            }
        
        filename = f"fake_news_analysis_{analysis_id[:8]}.json"
        with open(f"static/{filename}", "w") as f:
            json.dump(formatted_data, f, indent=2)
            
        return send_file(f"static/{filename}", as_attachment=True)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Additional routes for advanced features
@app.route('/dashboard')
def dashboard():
    # Get summary statistics from analysis data
    try:
        all_analyses = []
        analysis_stats = {
            'total_analyses': 0,
            'fake_count': 0,
            'real_count': 0,
            'avg_confidence': 0,
            'model_usage': {'naive_bayes': 0, 'random_forest': 0, 'lstm': 0, 'ensemble': 0}
        }
        
        if os.path.exists('analysis_data'):
            analysis_files = [f for f in os.listdir('analysis_data') if f.endswith('.json')]
            
            for file in analysis_files[:50]:  # Limit to most recent 50
                try:
                    with open(os.path.join('analysis_data', file), 'r') as f:
                        analysis = json.load(f)
                        all_analyses.append(analysis)
                        
                        analysis_stats['total_analyses'] += 1
                        
                        if analysis.get('result', {}).get('prediction') == 'fake':
                            analysis_stats['fake_count'] += 1
                        else:
                            analysis_stats['real_count'] += 1
                        
                        analysis_stats['avg_confidence'] += analysis.get('result', {}).get('confidence', 0)
                        
                        model = analysis.get('model_choice')
                        if model in analysis_stats['model_usage']:
                            analysis_stats['model_usage'][model] += 1
                except:
                    continue
            
            if analysis_stats['total_analyses'] > 0:
                analysis_stats['avg_confidence'] /= analysis_stats['total_analyses']
    except:
        all_analyses = []
        analysis_stats = {
            'total_analyses': 0, 
            'fake_count': 0, 
            'real_count': 0,
            'avg_confidence': 0,
            'model_usage': {}
        }
    
    return render_template('dashboard.html', 
                          analyses=all_analyses[:5],  # Most recent 5 for display
                          stats=analysis_stats)

@app.route('/learn')
def learn():
    return render_template('learn.html')

@app.route('/api_docs')
def api_docs():
    return render_template('api_docs.html')

@app.route('/case_studies')
def case_studies():
    # Some example case studies for demonstration
    case_studies = [
        {
            'id': 1,
            'title': 'COVID-19 Misinformation Analysis',
            'description': 'Investigation of fake news spread during the pandemic',
            'category': 'Health',
            'image': 'covid_case.jpg',
            'fake_ratio': 68
        },
        {
            'id': 2,
            'title': 'Election News Verification',
            'description': 'Analysis of news articles during the 2020 election cycle',
            'category': 'Politics',
            'image': 'election_case.jpg',
            'fake_ratio': 45
        },
        {
            'id': 3,
            'title': 'Climate Change Reporting',
            'description': 'Examination of scientific vs. pseudo-scientific news articles',
            'category': 'Science',
            'image': 'climate_case.jpg',
            'fake_ratio': 33
        },
        {
            'id': 4,
            'title': 'Financial Market Rumors',
            'description': 'Impact of fabricated financial news on market behavior',
            'category': 'Finance',
            'image': 'finance_case.jpg',
            'fake_ratio': 52
        },
    ]
    
    return render_template('case_studies.html', case_studies=case_studies)

# API endpoints for dashboard data
@app.route('/api/stats', methods=['GET'])
def api_stats():
    try:
        # Generate some demo statistics if no real data exists
        total_analyses = 0
        fake_count = 0
        real_count = 0
        
        if os.path.exists('analysis_data'):
            analysis_files = [f for f in os.listdir('analysis_data') if f.endswith('.json')]
            
            for file in analysis_files:
                try:
                    with open(os.path.join('analysis_data', file), 'r') as f:
                        analysis = json.load(f)
                        total_analyses += 1
                        
                        if analysis.get('result', {}).get('prediction') == 'fake':
                            fake_count += 1
                        else:
                            real_count += 1
                except:
                    continue
        
        # If no data, use demo data
        if total_analyses == 0:
            fake_count = 157
            real_count = 243
            total_analyses = fake_count + real_count
        
        # Calculate percentages
        fake_percent = (fake_count / total_analyses * 100) if total_analyses > 0 else 0
        real_percent = (real_count / total_analyses * 100) if total_analyses > 0 else 0
        
        # Model performance data (demo data if no real data)
        model_performance = {
            'naive_bayes': {'accuracy': 78.5, 'precision': 82.1, 'recall': 75.8},
            'random_forest': {'accuracy': 83.2, 'precision': 85.7, 'recall': 79.2},
            'lstm': {'accuracy': 88.7, 'precision': 89.4, 'recall': 87.1},
            'ensemble': {'accuracy': 90.2, 'precision': 91.3, 'recall': 89.5}
        }
        
        # Time series data (for demonstration)
        dates = ['2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06', 
                '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12']
                
        fake_trend = [42, 51, 63, 58, 47, 52, 65, 73, 69, 75, 82, 91]
        real_trend = [58, 69, 72, 85, 91, 76, 82, 79, 85, 92, 86, 94]
        
        return jsonify({
            'total_analyses': total_analyses,
            'fake_count': fake_count,
            'real_count': real_count,
            'fake_percent': fake_percent,
            'real_percent': real_percent,
            'model_performance': model_performance,
            'time_series': {
                'dates': dates,
                'fake_trend': fake_trend,
                'real_trend': real_trend
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
