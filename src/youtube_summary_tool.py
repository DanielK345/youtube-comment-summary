"""
Main orchestration file for YouTube comment analysis.

This module coordinates all components:
- YouTube API data fetching
- ChromaDB storage and retrieval
- Sentiment analysis
- Summary generation
- Command-line interface
"""

import argparse
import os
import sys
import logging
from dotenv import load_dotenv

# Add parent directory to path to access utils from root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load .env file from project root
load_dotenv(os.path.join(project_root, '.env'))

# Import all necessary functions and constants from refactored modules
from src.utils import (
    extract_video_id,
    get_comments,
    preprocess_for_sentiment,
    preprocess_comment_for_wordcloud,
    plot_sentiment_pie_chart,
    generate_wordcloud,
    setup_logger,
    YouTubeClient
)

from src.chroma_db import (
    CHROMA_PATH,
    CURRENT_VIDEO_ID,
    close_chroma_connection,
    get_embedding_function,
    get_chroma_db,
    save_comments_to_chroma,
    read_comments_from_chroma,
    calculate_optimal_k,
    answer_question,
    generate_comment_summary,
    get_current_video_id,
    get_chroma_path,
    ChromaDBManager
)

from src.models import (
    MODEL_NAME,
    LABEL_MAPPING,
    get_tokenizer,
    get_hf_model,
    get_language_model,
    SentimentAnalyzer,
    SentimentVectorStore,
    ModelManager
)


class YouTubeAnalyzer:
    """Main orchestrator for YouTube comment analysis."""
    
    def __init__(self, youtube_api_key: str = None, chroma_path: str = "chroma", logger: logging.Logger = None):
        """
        Initialize YouTubeAnalyzer.
        
        Args:
            youtube_api_key: YouTube API key (uses default if not provided)
            chroma_path: Path for ChromaDB storage
            logger: Optional logger instance
        """
        self.logger = logger or setup_logger(__name__)
        self.youtube_api_key = youtube_api_key or "AIzaSyDj7I12G6kpxEt4esWYXh2XwVAOXu7mbz0"
        
        # Initialize components
        try:
            self.logger.info("Initializing YouTubeAnalyzer components...")
            self.youtube_client = YouTubeClient(self.youtube_api_key, logger=self.logger)
            self.chroma_manager = ChromaDBManager(chroma_path=chroma_path, logger=self.logger)
            self.model_manager = ModelManager(logger=self.logger)
            self.logger.info("YouTubeAnalyzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize YouTubeAnalyzer: {e}", exc_info=True)
            raise
    
    def _create_error_response(self, video_id: str, error_msg: str, comment_count: int = 0):
        """Create a standardized error response."""
        return {
            "video_id": video_id,
            "comment_count": comment_count,
            "error": error_msg,
            "output_files": {
                "sentiment_chart": None,
                "wordcloud": None,
                "overall_summary": None,
                "sentiment_summary": None
            }
        }
    
    def analyze_youtube_comments(self, youtube_url: str):
        """
        Main function to analyze YouTube comments from a URL.

        Args:
            youtube_url: URL or video ID of the YouTube video

        Returns:
            Dictionary with summaries and analysis results
        """
        self.logger.info(f"Starting analysis for: {youtube_url}")

        # Create the main Chroma directory if it doesn't exist
        os.makedirs(self.chroma_manager.chroma_path, exist_ok=True)

        # Extract video ID if full URL is provided
        video_id = extract_video_id(youtube_url)
        if not video_id:
            error_msg = "Invalid YouTube URL or video ID"
            self.logger.error(error_msg)
            return self._create_error_response(None, error_msg)

        self.logger.info(f"Extracted video ID: {video_id}")

        # Step 1: Get comments from YouTube API
        try:
            self.logger.info("Step 1: Fetching comments from YouTube...")
            comments = self.youtube_client.get_comments(video_id)

            if not comments:
                error_msg = "No comments found or comments are disabled for this video"
                self.logger.warning(error_msg)
                return self._create_error_response(video_id, error_msg)
        except Exception as e:
            error_msg = f"Error fetching comments: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return self._create_error_response(video_id, error_msg)

        # Step 2: Save comments to Chroma vector database
        try:
            self.logger.info("Step 2: Saving comments to vector database...")
            comment_count = self.chroma_manager.save_comments_to_chroma(comments, video_id)
        except Exception as e:
            error_msg = f"Error saving comments to database: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return self._create_error_response(video_id, error_msg, len(comments))

        # Step 3: Read comments from Chroma
        try:
            self.logger.info("Step 3: Reading comments from database...")
            raw_comments = self.chroma_manager.read_comments_from_chroma(video_id)
            if not raw_comments:
                error_msg = "Could not retrieve comments from database"
                self.logger.error(error_msg)
                return self._create_error_response(video_id, error_msg, comment_count)
        except Exception as e:
            error_msg = f"Error reading comments from database: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return self._create_error_response(video_id, error_msg, comment_count)

        # Step 4: Generate overall comment summary
        overall_summary = None
        try:
            self.logger.info("Step 4: Generating overall comment summary...")
            overall_summary = self.chroma_manager.generate_comment_summary(video_id)
        except Exception as e:
            error_msg = f"Error generating summary: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            overall_summary = f"Error generating summary: {str(e)}"

        # Step 5: Preprocess comments for sentiment analysis
        sentiment_summary = {"error": "Not processed"}
        pos_summary = "Error generating positive summary"
        neg_summary = "Error generating negative summary"
        chart_path = None
        wordcloud_path = None
        summary_path = None

        try:
            self.logger.info("Step 5: Analyzing sentiment...")
            processed_comments = preprocess_for_sentiment(raw_comments)

            # Step 6: Perform sentiment analysis
            self.logger.info("Step 6: Performing sentiment analysis...")
            sentiment_analyzer = SentimentAnalyzer(model_manager=self.model_manager, logger=self.logger)
            sentiment_results, positive_comments, negative_comments, neutral_comments = sentiment_analyzer.analyze_sentiment(
                processed_comments)

            # Step 7: Create sentiment visualization
            self.logger.info("Step 7: Creating visualizations...")
            sentiment_chart = plot_sentiment_pie_chart(sentiment_results)

            # Save in video-specific directory
            chart_dir = os.path.join(self.chroma_manager.chroma_path, video_id)
            os.makedirs(chart_dir, exist_ok=True)
            chart_path = os.path.join(chart_dir, "sentiment_pie_chart.png")
            sentiment_chart.savefig(chart_path)
            self.logger.info(f"Sentiment chart saved to {chart_path}")

            # Step 8: Generate word cloud
            self.logger.info("Step 8: Generating word cloud...")
            wordcloud = generate_wordcloud(raw_comments)
            wordcloud_path = os.path.join(chart_dir, "comment_wordcloud.png")
            wordcloud.savefig(wordcloud_path)
            self.logger.info(f"Word cloud saved to {wordcloud_path}")

            # Step 9: Summarize positive and negative comments
            self.logger.info("Step 9: Generating sentiment-specific summaries...")
            summary_path = os.path.join(chart_dir, "sentiment_summary.txt")
            sentiment_store = SentimentVectorStore(model_manager=self.model_manager, logger=self.logger)
            pos_summary, neg_summary = sentiment_store.summarize_both_sentiments(
                positive_comments, negative_comments, output_file=summary_path)

            # Save output files in video-specific directory
            sentiment_summary = {
                "positive": len(positive_comments),
                "negative": len(negative_comments),
                "neutral": len(neutral_comments)
            }
            self.logger.info("Sentiment analysis completed successfully")
        except Exception as e:
            error_msg = f"Error in sentiment analysis: {e}"
            self.logger.error(error_msg, exc_info=True)
            sentiment_summary = {"error": str(e)}

        # Return results
        results = {
            "video_id": video_id,
            "comment_count": comment_count,
            "overall_summary": overall_summary,
            "sentiment_counts": sentiment_summary,
            "positive_summary": pos_summary,
            "negative_summary": neg_summary,
            "output_files": {
                "sentiment_chart": chart_path,
                "wordcloud": wordcloud_path,
                "overall_summary": os.path.join(self.chroma_manager.chroma_path, video_id, "overall_summary.txt"),
                "sentiment_summary": summary_path
            }
        }

        self.logger.info("Analysis complete! Results saved to output files.")
        return results


# Backward compatibility function
def analyze_youtube_comments(youtube_url, api_key="AIzaSyDj7I12G6kpxEt4esWYXh2XwVAOXu7mbz0"):
    """
    Main function to analyze YouTube comments from a URL (backward compatibility wrapper).

    Args:
        youtube_url: URL or video ID of the YouTube video
        api_key: YouTube API key (uses default if not provided)

    Returns:
        Dictionary with summaries and analysis results
    """
    analyzer = YouTubeAnalyzer(youtube_api_key=api_key)
    return analyzer.analyze_youtube_comments(youtube_url)


# For backward compatibility, create wrapper functions that match old API
def analyze_sentiment(comments):
    """Wrapper for backward compatibility."""
    analyzer = SentimentAnalyzer()
    return analyzer.analyze_sentiment(comments)


def summarize_both_sentiments(positive_comments, negative_comments, output_file="sentiment_summary.txt"):
    """Wrapper for backward compatibility."""
    store = SentimentVectorStore()
    return store.summarize_both_sentiments(positive_comments, negative_comments, output_file)


# Re-export for backward compatibility
__all__ = [
    # Constants
    'CHROMA_PATH',
    'CURRENT_VIDEO_ID',
    'MODEL_NAME',
    'LABEL_MAPPING',
    
    # Utility functions
    'extract_video_id',
    'get_comments',
    'preprocess_for_sentiment',
    'preprocess_comment_for_wordcloud',
    'plot_sentiment_pie_chart',
    'generate_wordcloud',
    
    # ChromaDB functions
    'close_chroma_connection',
    'get_embedding_function',
    'get_chroma_db',
    'save_comments_to_chroma',
    'read_comments_from_chroma',
    'calculate_optimal_k',
    'answer_question',
    'generate_comment_summary',
    'get_current_video_id',
    'get_chroma_path',
    
    # Model functions
    'get_tokenizer',
    'get_hf_model',
    'get_language_model',
    'SentimentAnalyzer',
    'SentimentVectorStore',
    
    # Main orchestration
    'analyze_youtube_comments',
    'analyze_sentiment',
    'summarize_both_sentiments',
    'YouTubeAnalyzer',
]


# Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YouTube Comment Analysis Tool")
    parser.add_argument(
        "youtube_url", help="YouTube URL or video ID to analyze")
    parser.add_argument("--api-key", default="AIzaSyDj7I12G6kpxEt4esWYXh2XwVAOXu7mbz0",
                        help="YouTube API key (optional)")
    parser.add_argument(
        "--question", help="Ask a specific question about the comments")
    parser.add_argument("--k", type=int, default=None,
                        help="Number of comments to retrieve for context (default: auto-calculated)")
    parser.add_argument("--reuse-db", action="store_true",
                        help="Force reuse of existing database without confirmation")

    args = parser.parse_args()

    # Extract video ID
    video_id = extract_video_id(args.youtube_url)

    # Check if we need to run the analysis
    run_analysis = True
    current_video_id = get_current_video_id()
    chroma_path = get_chroma_path()
    
    if current_video_id == video_id and os.path.exists(chroma_path) and args.reuse_db:
        print(f"Using existing analysis for video ID: {video_id}")
        run_analysis = False

    # Initialize analyzer
    analyzer = YouTubeAnalyzer(youtube_api_key=args.api_key)

    # Check if a question was asked
    if args.question:
        # First make sure we've analyzed the comments
        if run_analysis:
            print(f"Analyzing comments for: {args.youtube_url}")
            analyzer.analyze_youtube_comments(args.youtube_url)

        # Then answer the question
        print(f"\nQuestion: {args.question}")
        print("\nSearching for answer...")
        answer = analyzer.chroma_manager.answer_question(args.question, k=args.k, video_id=video_id)
        print("\nAnswer:")
        print(answer)
    else:
        # Regular analysis
        if run_analysis:
            results = analyzer.analyze_youtube_comments(args.youtube_url)

            # Print a summary of results
            print("\n===== ANALYSIS RESULTS =====")
            print(f"Video ID: {results['video_id']}")
            print(f"Total comments analyzed: {results['comment_count']}")
            
            if 'error' not in results.get('sentiment_counts', {}):
                print(f"Sentiment distribution: {results['sentiment_counts']['positive']} positive, "
                      f"{results['sentiment_counts']['negative']} negative, "
                      f"{results['sentiment_counts']['neutral']} neutral")
            
            print("Output files:")
            for name, path in results['output_files'].items():
                if path:
                    print(f"- {name}: {path}")
        else:
            print("To perform a new analysis, run without the --reuse-db flag.")
            print("To ask a question about the existing analysis, use the --question parameter.")
