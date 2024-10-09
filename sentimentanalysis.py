import praw
import spacy
from textblob import TextBlob
import pandas as pd

# Step 1: Data Collection
# Initialize the Reddit API client
reddit = praw.Reddit(client_id='6e3Op2hNfb4bdfT2B5CovQ',
                     client_secret='rs6rYv7NJSL7mxt8X-fZ7Vs-vNXaLA',
                     user_agent='sentimentanalyser/1.0 by alina')

# Define the subreddit to scrape
subreddit = reddit.subreddit('mentalhealth')

# Collect posts
posts = []
for submission in subreddit.new(limit=100):  # Adjust limit as needed
    posts.append(submission.title + " " + submission.selftext)

# Step 2: Text Preprocessing
# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Preprocess the text data
def preprocess_text(text):
    doc = nlp(text)
    # Remove stop words and punctuation, and return lemmatized text
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Apply preprocessing to all posts
cleaned_posts = [preprocess_text(post) for post in posts]

# Step 3: Sentiment Analysis
# Perform sentiment analysis
sentiments = [TextBlob(post).sentiment.polarity for post in cleaned_posts]

# Classify sentiments
def classify_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

sentiment_classes = [classify_sentiment(score) for score in sentiments]

# Step 4: Create DataFrame for Analysis
data = {
    'text': cleaned_posts,  # Use cleaned_posts
    'sentiment': sentiment_classes
}
df = pd.DataFrame(data)

# Optionally, you can print or analyze the DataFrame
print(df.head())

