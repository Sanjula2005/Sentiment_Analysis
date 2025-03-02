# Sentiment Analysis Web Application

## Overview
This project is a **Sentiment Analysis Web Application** built using **Flask**, **NLTK**, and **Scikit-Learn**. The application takes user input (text reviews), processes the text, and classifies the sentiment as **Positive, Neutral, or Negative** using a **Logistic Regression** model trained on customer review data.

## Features
- Web-based interface for user-friendly sentiment analysis
- Preprocessing of text using NLTK (stopword removal, lemmatization)
- TF-IDF vectorization for feature extraction
- Sentiment classification using a trained Logistic Regression model
- REST API endpoint (`/analyze`) for sentiment prediction

## Tech Stack
- **Backend:** Flask, Pandas, NLTK, Scikit-Learn
- **Frontend:** HTML, CSS, JavaScript
- **Machine Learning:** Logistic Regression, TF-IDF Vectorizer



## File Structure
```
Sentiment_Analysis/
│── static/
│── templates/
│   ├── index.html  # Frontend UI
│── app.py  # Main Flask Application
│── Reviews.csv  # Dataset
│── requirements.txt  # Dependencies
```

## Future Enhancements
- Improve the model with deep learning techniques (LSTMs, Transformers)
- Add support for multilingual sentiment analysis


