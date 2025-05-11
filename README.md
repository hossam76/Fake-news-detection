# Fake News Detection System 🛡️

![Fake News Detection](https://img.shields.io/badge/ML-Fake%20News%20Detection-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Flask](https://img.shields.io/badge/Flask-2.0.1-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-success)

A sophisticated web application built to detect and analyze fake news using machine learning algorithms. This project leverages multiple ML models including Naive Bayes, Random Forest, and LSTM to provide comprehensive analysis with explanations and visualizations.

## 🌟 Features

- **Multi-model Fake News Detection**: Uses Naive Bayes, Random Forest, and LSTM models
- **Ensemble Prediction**: Combines results from all models for more accurate detection
- **Advanced Text Analysis**: Analyzes sentiment, entities, word frequency, and more
- **Visualizations**: Interactive dashboard with charts and wordclouds
- **URL Processing**: Extract and analyze news directly from URLs
- **API Integration**: RESTful API for integration with other services
- **Explainable AI**: Uses LIME to explain why content was classified as fake or real
- **Source Credibility Checking**: Assesses reliability of news sources
- **Detailed Case Studies**: In-depth analyses of fake news in various domains

## 🔧 Tech Stack

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn, TensorFlow, NLTK, spaCy
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Chart.js
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Explanatory AI**: LIME
- **Web Scraping**: Newspaper3k, BeautifulSoup

## 📋 Prerequisites

- Python 3.8+
- pip package manager
- Virtual environment (recommended)

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hossam76/Fake-news-detection.git
   cd Fake-news-detection
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   #if need to upgrade pip-- python -m pip install --upgrade pip

   ```

4. Download NLTK resources (or let the app download them on first run):
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
   ```

5. Download spaCy model (optional but recommended):
   ```bash
   python -m spacy download en_core_web_sm
   ```

6. Create the mock LSTM model (for testing if you don't have the trained model):
   ```bash
   python create_mock_lstm.py
   ```
7. need to install tensor flow and newspaper3k
   ```bash
   pip install tensorflow
   pip install newspaper3k
   ```

## 💻 Usage

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Enter news text or a URL to analyze for fake news.

4. View the analysis results, including:
   - Prediction (real or fake)
   - Confidence score
   - Detailed text analysis
   - Word cloud visualization
   - Source credibility (if URL provided)
   - Explanation of the prediction

## 📊 Models

This system uses three main models for fake news detection:

1. **Naive Bayes**: Fast and efficient text classification
2. **Random Forest**: More complex analysis with feature importance
3. **LSTM**: Deep learning approach for sequence analysis
4. **Ensemble**: Combination of all models for higher accuracy

The models were trained on a labeled dataset of real and fake news articles, achieving:
- Naive Bayes: ~85% accuracy
- Random Forest: ~88% accuracy
- LSTM: ~92% accuracy
- Ensemble: ~95% accuracy

## 📚 Dataset

The models were trained using the "Fake News Detection Datasets" available on Kaggle:

**Dataset URL**: [Fake News Detection Datasets on Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data)

**Note**: Due to storage constraints, the dataset is not included in this repository. To train the models from scratch, please:

1. Download the dataset from the Kaggle link above
2. Place the CSV files (`True.csv` and `Fake.csv`) in the project root directory
3. Run the preprocessing and training notebook: `Models Trannings.ipynb`

The dataset contains thousands of labeled news articles categorized as either "real" or "fake", providing a robust foundation for training our detection models.

## 🔎 API Reference

The system provides a REST API for integration:

- `POST /predict`: Analyze text with a specific model
- `POST /predict_ensemble`: Analyze text with the ensemble model
- `POST /analyze_url`: Extract and analyze content from a URL
- `GET /api/stats`: Retrieve analysis statistics

Example:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news text here", "model": "ensemble"}'
```

## 📂 Project Structure

