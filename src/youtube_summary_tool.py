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

# Add parent directory to path to access utils from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all necessary functions and constants from refactored modules
from utils import (
    extract_video_id,
    get_comments,
    preprocess_for_sentiment,
    preprocess_comment_for_wordcloud,
    plot_sentiment_pie_chart,
    generate_wordcloud
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
    get_chroma_path
)

from src.models import (
    MODEL_NAME,
    LABEL_MAPPING,
    get_tokenizer,
    get_hf_model,
    get_language_model,
    SentimentAnalyzer,
    SentimentVectorStore
)


def analyze_youtube_comments(youtube_url, api_key="AIzaSyDj7I12G6kpxEt4esWYXh2XwVAOXu7mbz0"):
    """
    Main function to analyze YouTube comments from a URL.

    Args:
        youtube_url: URL or video ID of the YouTube video
        api_key: YouTube API key (uses default if not provided)

    Returns:
        Dictionary with summaries and analysis results
    """
    print(f"Analyzing comments for: {youtube_url}")

    # Create the main Chroma directory if it doesn't exist
    os.makedirs(CHROMA_PATH, exist_ok=True)

    # Extract video ID if full URL is provided
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return {
            "error": "Invalid YouTube URL or video ID"
        }

    print(f"Extracted video ID: {video_id}")

    # Step 1: Get comments from YouTube API
    try:
        print("Fetching comments from YouTube...")
        comments = get_comments(video_id, api_key)

        if not comments:
            return {
                "video_id": video_id,
                "error": "No comments found or comments are disabled for this video"
            }
    except Exception as e:
        return {
            "video_id": video_id,
            "error": f"Error fetching comments: {str(e)}"
        }

    # Step 2: Save comments to Chroma vector database
    try:
        print("Saving comments to vector database...")
        comment_count = save_comments_to_chroma(comments, video_id)
    except Exception as e:
        return {
            "video_id": video_id,
            "error": f"Error saving comments to database: {str(e)}"
        }

    # Step 3: Read comments from Chroma
    try:
        raw_comments = read_comments_from_chroma(video_id)
        if not raw_comments:
            return {
                "video_id": video_id,
                "comment_count": comment_count,
                "error": "Could not retrieve comments from database"
            }
    except Exception as e:
        return {
            "video_id": video_id,
            "comment_count": comment_count,
            "error": f"Error reading comments from database: {str(e)}"
        }

    # Step 4: Generate overall comment summary
    try:
        print("Generating overall comment summary...")
        overall_summary = generate_comment_summary(video_id)
    except Exception as e:
        overall_summary = f"Error generating summary: {str(e)}"
        print(overall_summary)

    # Step 5: Preprocess comments for sentiment analysis
    try:
        print("Analyzing sentiment...")
        processed_comments = preprocess_for_sentiment(raw_comments)

        # Step 6: Perform sentiment analysis
        sentiment_analyzer = SentimentAnalyzer()
        sentiment_results, positive_comments, negative_comments, neutral_comments = sentiment_analyzer.analyze_sentiment(
            processed_comments)

        # Step 7: Create sentiment visualization
        print("Creating visualizations...")
        sentiment_chart = plot_sentiment_pie_chart(sentiment_results)

        # Save in video-specific directory
        chart_dir = os.path.join(CHROMA_PATH, video_id)
        os.makedirs(chart_dir, exist_ok=True)
        chart_path = os.path.join(chart_dir, "sentiment_pie_chart.png")
        sentiment_chart.savefig(chart_path)

        # Step 8: Generate word cloud
        wordcloud = generate_wordcloud(raw_comments)
        wordcloud_path = os.path.join(chart_dir, "comment_wordcloud.png")
        wordcloud.savefig(wordcloud_path)

        # Step 9: Summarize positive and negative comments
        print("Generating sentiment-specific summaries...")
        summary_path = os.path.join(chart_dir, "sentiment_summary.txt")
        sentiment_store = SentimentVectorStore()
        pos_summary, neg_summary = sentiment_store.summarize_both_sentiments(
            positive_comments, negative_comments, output_file=summary_path)

        # Save output files in video-specific directory
        sentiment_summary = {
            "positive": len(positive_comments),
            "negative": len(negative_comments),
            "neutral": len(neutral_comments)
        }
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        sentiment_summary = {"error": str(e)}
        pos_summary = "Error generating positive summary"
        neg_summary = "Error generating negative summary"
        chart_path = None
        wordcloud_path = None
        summary_path = None

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
            "overall_summary": os.path.join(CHROMA_PATH, video_id, "overall_summary.txt"),
            "sentiment_summary": summary_path
        }
    }

    print("\nAnalysis complete! Results saved to output files.")
    return results


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

    # Check if a question was asked
    if args.question:
        # First make sure we've analyzed the comments
        if run_analysis:
            print(f"Analyzing comments for: {args.youtube_url}")
            analyze_youtube_comments(args.youtube_url, args.api_key)

        # Then answer the question
        print(f"\nQuestion: {args.question}")
        print("\nSearching for answer...")
        answer = answer_question(args.question, k=args.k, video_id=video_id)
        print("\nAnswer:")
        print(answer)
    else:
        # Regular analysis
        if run_analysis:
            results = analyze_youtube_comments(args.youtube_url, args.api_key)

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
