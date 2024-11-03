from flask import Flask, jsonify, request
from flask_cors import CORS
import nltk
import string
from heapq import nlargest
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer

# Download necessary nltk resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
CORS(app)

# Preprocessing and Summarization Functions
def preprocess_text(text):
    # Lowercase and remove punctuation
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])

    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in word_tokenize(text) if word not in stop_words]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return words

def score_sentences(sentences, word_scores):
    sent_scores = {}
    for idx, sent in enumerate(sentences):
        words = preprocess_text(sent)
        sentence_score = sum(word_scores.get(word, 0) for word in words)
        
        # Boost score for first and last sentence positions
        if idx == 0 or idx == len(sentences) - 1:
            sentence_score *= 1.2

        # Filter out short sentences
        if len(words) > 3:
            sent_scores[sent] = sentence_score
    return sent_scores

def summaryFunc(text):
    sentences = sent_tokenize(text)

    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(sentences)
    feature_names = tfidf.get_feature_names_out()

    # Word scores using TF-IDF
    word_scores = {feature_names[i]: tfidf_matrix.getcol(i).sum() for i in range(len(feature_names))}

    # Score sentences and select top ones
    sent_scores = score_sentences(sentences, word_scores)
    summary_length = max(1, int(len(sentences) / 4))
    summary_sentences = nlargest(summary_length, sent_scores, key=sent_scores.get)
    
    # Join selected sentences into the final summary
    summary = " ".join(summary_sentences)
    return summary

def calculate_rouge(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

@app.route('/predict', methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get('passage', '')
    reference_summary = data.get('reference', '')  # Reference summary provided by the user

    # Generate summary
    summary = summaryFunc(text)

    # Calculate ROUGE score if reference is provided
    rouge_scores = calculate_rouge(reference_summary, summary) if reference_summary else None

    return jsonify({
        "summary": summary,
        "rouge_scores": rouge_scores
    })

if __name__ == '__main__':
    app.run(debug=True)
