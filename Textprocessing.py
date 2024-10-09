import praw
import spacy
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora
from gensim.models import LdaModel
import matplotlib.pyplot as plt
import seaborn as sns
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

# Save posts to a text file for later use
with open('mental_health_posts.txt', 'w', encoding='utf-8') as f:
    for post in posts:
        f.write(post + '\n')

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

# Step 4: Prepare DataFrame for Classification
# Ensure we only use data where sentiments were successfully classified
data = {
    'text': cleaned_posts,
    'sentiment': sentiment_classes
}
df = pd.DataFrame(data)

# Check lengths of text and sentiment lists
print("Length of cleaned_posts:", len(cleaned_posts))
print("Length of sentiment_classes:", len(sentiment_classes))

# Step 5: Text Classification (Optional)
# Ensure the lengths match before proceeding
if len(cleaned_posts) == len(sentiment_classes):
    # Replace 'category' with actual labels if available
    # Example: Create a placeholder category list (ensure it matches the length of cleaned_posts)
    categories = ['anxiety', 'depression', 'wellness', 'anxiety', 'depression'] * (len(df) // 5 + 1)  # Repeat to ensure length
    categories = categories[:len(df)]  # Trim to exact length

    df['category'] = categories  # Add categories to the DataFrame

    # Vectorize the text
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['category']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    # Predict on test set
    predictions = classifier.predict(X_test)

else:
    print("Length mismatch: Cannot create DataFrame for classification.")

# Step 6: Topic Modeling
# Create a dictionary and corpus for topic modeling
dictionary = corpora.Dictionary([post.split() for post in cleaned_posts])
corpus = [dictionary.doc2bow(post.split()) for post in cleaned_posts]

# Train the LDA model
lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# Print the topics found by the model
for idx, topic in lda_model.print_topics(-1):
    print(f'Topic {idx}: {topic}')

# Step 7: Visualization
# Sentiment distribution visualization
plt.figure(figsize=(10, 6))
sns.countplot(x=sentiment_classes)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()
