import praw

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
