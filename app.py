from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Initialize Flask App
app = Flask(__name__)

# Load and preprocess dataset
file_path = "Reviews.csv"
df = pd.read_csv(file_path)

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

df['Cleaned_Text'] = df['Summary'].astype(str).apply(clean_text)

# Convert text to features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Cleaned_Text'])

# Assign sentiment labels
def assign_sentiment(score):
    if score >= 4:
        return 1  # Positive
    elif score == 3:
        return 0  # Neutral
    else:
        return -1  # Negative

df['Sentiment'] = df['Score'].apply(assign_sentiment)

# Train-test split
y = df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    review = data.get('text', '')
    clean_review = clean_text(review)
    vectorized_review = vectorizer.transform([clean_review])
    sentiment = model.predict(vectorized_review)[0]

    sentiment_map = {1: "Positive", 0: "Neutral", -1: "Negative"}
    return jsonify({'sentiment': sentiment_map[sentiment]})

if __name__ == '__main__':
    app.run(debug=True)
