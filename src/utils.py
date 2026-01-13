"""
Utility functions for YouTube comment analysis.

This module handles:
- YouTube API interactions
- Text preprocessing
- Visualization functions
- Helper utilities
"""

import re
import time
import os
import nltk
from unidecode import unidecode
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from googleapiclient import discovery
import matplotlib
import matplotlib.pyplot as plt

# Set matplotlib backend before importing anything matplotlib-related
matplotlib.use('Agg')
plt.ioff()  # Turn off interactive mode

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)


def extract_video_id(youtube_url):
    """
    Extract video ID from a YouTube URL.
    
    Args:
        youtube_url: YouTube URL or video ID string
        
    Returns:
        Video ID string or None if not found
    """
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
    ]

    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)

    # If no patterns match, assume it's already a video ID if it's 11 chars
    if len(youtube_url) == 11:
        return youtube_url

    return None


def get_comments(video_id, api_key):
    """
    Fetch comments and replies from YouTube with improved data structure.

    Args:
        video_id: YouTube video ID
        api_key: YouTube API key

    Returns:
        List of comment dictionaries with author, text, likes, etc.
    """
    # Create a YouTube API client
    youtube = discovery.build('youtube', 'v3', developerKey=api_key)

    # Call the API to get the comments
    comments = []
    next_page_token = None
    total_comments = 0

    print("Fetching comments from YouTube API...")

    try:
        while True:
            # Request comments
            request = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                pageToken=next_page_token,
                maxResults=100,  # Maximum allowed by API
                textFormat='plainText'
            )
            response = request.execute()

            # Handle potential API errors
            if 'error' in response:
                print(f"API Error: {response['error']['message']}")
                break

            # Extract top-level comments and replies
            items_count = len(response.get('items', []))
            if items_count == 0:
                print("No comments found or all comments processed.")
                break

            print(f"Processing batch of {items_count} comment threads...")

            for item in response.get('items', []):
                # Top-level comment
                top_level_comment = item['snippet']['topLevelComment']['snippet']
                comment = top_level_comment['textDisplay']
                author = top_level_comment['authorDisplayName']
                likes = top_level_comment.get('likeCount', 0)

                # No 'replied_to' for top-level comment
                comments.append({
                    'author': author,
                    'comment': comment,
                    'likes': likes
                })
                total_comments += 1

                # Replies (if any)
                if 'replies' in item:
                    for reply in item['replies']['comments']:
                        reply_author = reply['snippet']['authorDisplayName']
                        reply_comment = reply['snippet']['textDisplay']
                        reply_likes = reply['snippet'].get('likeCount', 0)

                        # Include the 'replied_to' field only for replies
                        comments.append({
                            'author': reply_author,
                            'comment': reply_comment,
                            'replied_to': author,
                            'likes': reply_likes
                        })
                        total_comments += 1

            # Print progress
            print(f"Fetched {total_comments} comments so far...")

            # Check for more comments (pagination)
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break  # No more pages, exit the loop

            # Add a small delay to avoid hitting API rate limits
            time.sleep(0.5)

    except Exception as e:
        print(f"Error fetching comments: {str(e)}")

    print(f"Completed fetching {total_comments} comments.")
    return comments


def preprocess_for_sentiment(comment_list):
    """
    Preprocess comments for sentiment analysis.
    
    Args:
        comment_list: List of comment strings
        
    Returns:
        List of preprocessed comment strings
    """
    processed_comments = []
    for comment in comment_list:
        # Remove URLs
        comment = re.sub(r'http\S+|www\S+|https\S+|t\.co\S+', '', comment)
        comment = unidecode(comment)
        # Clean excessive whitespace
        comment = re.sub(r'\s+', ' ', comment).strip()
        # Keep sentence as-is for sentiment analysis
        processed_comments.append(comment)
    return processed_comments


def preprocess_comment_for_wordcloud(comment_list):
    """
    Preprocess comments: remove stopwords and apply lemmatization.
    
    Args:
        comment_list: List of comment strings
        
    Returns:
        List of preprocessed comment strings
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    processed_comments = []
    for comment in comment_list:
        # Tokenize
        words = word_tokenize(comment)
        # Remove stopwords and lemmatize
        cleaned = [
            lemmatizer.lemmatize(word)
            for word in words
            if word.lower() not in stop_words and word.isalpha()
        ]
        # Join tokens back
        processed_comments.append(' '.join(cleaned))
    return processed_comments


def plot_sentiment_pie_chart(results):
    """
    Create a pie chart of sentiment distribution.
    
    Args:
        results: List of [neutral_count, positive_count, negative_count]
        
    Returns:
        matplotlib figure object
    """
    # Get the counts for each sentiment category
    num_neutral = results[0]
    num_positive = results[1]
    num_negative = results[2]

    labels = ['😊 Positive', '😠 Negative', '😐 Neutral']
    sizes = [num_positive, num_negative, num_neutral]
    colors = ['#DFF0D8', '#F2DEDE', '#EAEAEA']
    explode = (0.1, 0, 0)  # explode 1st slice (Positive)

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels,
           colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    plt.close(fig)  # Close the figure to prevent display
    return fig


def generate_wordcloud(all_comments):
    """
    Generate a word cloud from comments.
    
    Args:
        all_comments: List of comment strings
        
    Returns:
        matplotlib figure object
    """
    # Preprocess the entire list for word cloud
    processed_comments = preprocess_comment_for_wordcloud(all_comments)

    # Combine into a single string
    text_all = ' '.join(processed_comments)

    # Generate the word cloud
    wc_all = WordCloud(width=1000, height=500,
                       background_color='white').generate(text_all)

    # Create the figure and plot
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc_all, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    plt.close(fig)  # Close the figure to prevent display
    return fig
