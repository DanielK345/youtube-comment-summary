"""
Utility functions for YouTube comment analysis.

This module handles:
- Logging configuration
- YouTube API interactions
- Text preprocessing
- Visualization functions
- Helper utilities
"""

import re
import time
import os
import logging
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
# Newer NLTK versions may require `punkt_tab` for tokenization.
# Download it if available; ignore if the resource name doesn't exist.
try:
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)


def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up and configure a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        log_file: Optional path to log file
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


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


class YouTubeClient:
    """Handles YouTube API interactions for fetching comments."""
    
    def __init__(self, api_key: str, logger: logging.Logger = None):
        """
        Initialize YouTube API client.
        
        Args:
            api_key: YouTube API key
            logger: Optional logger instance
        """
        self.api_key = api_key
        self.logger = logger or setup_logger(__name__)
        self.youtube = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the YouTube API client."""
        try:
            self.youtube = discovery.build('youtube', 'v3', developerKey=self.api_key)
            self.logger.info("YouTube API client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize YouTube API client: {e}")
            raise
    
    def get_comments(self, video_id: str):
        """
        Fetch comments and replies from YouTube with improved data structure.

        Args:
            video_id: YouTube video ID

        Returns:
            List of comment dictionaries with author, text, likes, etc.
        """
        from tqdm import tqdm
        
        comments = []
        next_page_token = None
        total_comments = 0

        self.logger.info(f"Starting to fetch comments for video ID: {video_id}")

        try:
            # First, get an estimate of total comments (if possible)
            # We'll use a progress bar that updates as we fetch
            pbar = None
            pbar_active = False  # Flag to track if progress bar is active
            
            while True:
                # Request comments
                request = self.youtube.commentThreads().list(
                    part='snippet,replies',
                    videoId=video_id,
                    pageToken=next_page_token,
                    maxResults=100,  # Maximum allowed by API
                    textFormat='plainText'
                )
                response = request.execute()

                # Handle potential API errors
                if 'error' in response:
                    error_msg = response['error']['message']
                    self.logger.error(f"YouTube API Error: {error_msg}")
                    if pbar_active and pbar is not None:
                        try:
                            pbar.close()
                        except:
                            pass
                    raise RuntimeError(f"YouTube API Error: {error_msg}")

                # Extract top-level comments and replies
                items_count = len(response.get('items', []))
                if items_count == 0:
                    self.logger.info("No comments found or all comments processed.")
                    if pbar_active and pbar is not None:
                        try:
                            pbar.close()
                        except:
                            pass
                    break

                # Initialize progress bar on first iteration
                if not pbar_active:
                    pbar = tqdm(desc="Fetching comments", unit=" comments", dynamic_ncols=True)
                    pbar_active = True

                self.logger.debug(f"Processing batch of {items_count} comment threads...")

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
                    if pbar_active:
                        try:
                            pbar.update(1)
                        except:
                            pass

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
                            if pbar_active:
                                try:
                                    pbar.update(1)
                                except:
                                    pass

                # Check for more comments (pagination)
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    if pbar_active and pbar is not None:
                        try:
                            pbar.close()
                        except:
                            pass
                    break  # No more pages, exit the loop

                # Add a small delay to avoid hitting API rate limits
                time.sleep(0.5)

        except Exception as e:
            self.logger.error(f"Error fetching comments: {str(e)}", exc_info=True)
            if pbar_active and pbar is not None:
                try:
                    pbar.close()
                except:
                    pass
            raise

        self.logger.info(f"Completed fetching {total_comments} comments.")
        return comments


# Backward compatibility function
def get_comments(video_id, api_key):
    """
    Fetch comments and replies from YouTube (backward compatibility wrapper).
    
    Args:
        video_id: YouTube video ID
        api_key: YouTube API key
        
    Returns:
        List of comment dictionaries
    """
    client = YouTubeClient(api_key)
    return client.get_comments(video_id)


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
    # NOTE: Some `wordcloud` versions call `np.asarray(..., copy=...)`, which
    # requires NumPy 2.x. Your project pins NumPy 1.26.x, so convert via PIL
    # image to a NumPy array to avoid the incompatible `copy=` keyword.
    import numpy as np
    wc_img = np.array(wc_all.to_image())
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc_img, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    plt.close(fig)  # Close the figure to prevent display
    return fig
