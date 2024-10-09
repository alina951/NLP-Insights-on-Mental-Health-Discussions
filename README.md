# NLP-Insights-on-Mental-Health-Discussions

# Project Overview

This project analyses discussions on mental health from the Reddit subreddit mentalhealth. Using Natural Language Processing (NLP) techniques, the project aims to derive insights about sentiment, classification of topics, and visual representation of findings to better understand public discourse surrounding mental health issues.

# Features
Data collection from the Reddit API
Text preprocessing using spaCy
Sentiment analysis using TextBlob
Text classification using Random Forest Classifier
Topic modeling using LDA (Latent Dirichlet Allocation)
Visualization of sentiment distribution


Data Collection
The script collects the latest 100 posts from the mentalhealth subreddit, combining the title and body of each post. Collected data is saved to mental_health_posts.txt.

Text Preprocessing
The text data is preprocessed using spaCy to remove stop words and punctuation, followed by lemmatization to reduce words to their base forms.

Sentiment Analysis
Sentiment analysis is performed using TextBlob, which assigns a polarity score to each post, classifying it as Positive, Negative, or Neutral.

Text Classification
(Optional) The script includes a text classification component that categorizes posts into predefined categories (e.g., anxiety, depression, wellness) using a Random Forest classifier.

Topic Modeling
LDA is used to identify underlying topics in the posts. The model outputs the top topics and their associated words.

Visualisation
A count plot is generated to visualize the distribution of sentiments across the collected posts, allowing for a quick understanding of the general sentiment in the discussions.

Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

License
This project is licensed under the MIT License. See the LICENSE file for details.



