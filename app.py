import os
from dotenv import load_dotenv

# Load environment variables from .env file BEFORE any other imports
# Get the project root directory (where app.py is located)
project_root = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(project_root, '.env'))

from src.youtube_summary_tool import (
    extract_video_id, 
    CURRENT_VIDEO_ID, 
    YouTubeAnalyzer
)
from src.utils import setup_logger
from flask import Flask, request, jsonify, send_from_directory, render_template
import json
import time
import logging
import matplotlib

# Set matplotlib to use a non-interactive backend before any other matplotlib imports
matplotlib.use('Agg')

# Set up logging
logger = setup_logger(__name__, log_file=os.path.join(project_root, 'app.log'))

app = Flask(
    __name__,
    static_folder=os.path.join(project_root, 'static'),
    template_folder=os.path.join(project_root, 'templates')
)

# Lazy initialization - analyzer will be created on first use
# This speeds up server startup significantly
analyzer = None

def get_analyzer():
    """Get or create analyzer instance (lazy initialization)."""
    global analyzer
    if analyzer is None:
        try:
            logger.info("Initializing YouTubeAnalyzer (lazy load)...")
            analyzer = YouTubeAnalyzer(logger=logger)
            logger.info("YouTubeAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YouTubeAnalyzer: {e}", exc_info=True)
            raise
    return analyzer

# Store the latest analysis results
latest_results = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    global latest_results

    data = request.json
    youtube_url = data.get('youtube_url')

    if not youtube_url:
        logger.warning("No YouTube URL provided in analyze request")
        return jsonify({'error': 'No YouTube URL provided'}), 400

    try:
        # Get analyzer (lazy initialization)
        analyzer = get_analyzer()
        
        # Extract video ID to check if we're analyzing a new video
        video_id = extract_video_id(youtube_url)

        if not video_id:
            logger.warning(f"Invalid YouTube URL or video ID: {youtube_url}")
            return jsonify({'error': 'Invalid YouTube URL or video ID'}), 400

        logger.info(f"Starting analysis for video ID: {video_id}")

        # Make sure any previous connections are closed before analysis
        analyzer.chroma_manager.close_connection()

        # Run the analysis using the analyzer instance
        results = analyzer.analyze_youtube_comments(youtube_url)

        # Store the results for later use
        latest_results = results

        logger.info(f"Analysis completed successfully for video ID: {video_id}")
        return jsonify(results)
    except Exception as e:
        error_msg = f"Error during analysis: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({'error': error_msg}), 500


@app.route('/api/ask', methods=['POST'])
def ask_question_api():
    global latest_results

    data = request.json
    question = data.get('question')
    k = data.get('k')  # Optional parameter for number of comments to retrieve

    if not question:
        logger.warning("No question provided in ask request")
        return jsonify({'error': 'No question provided'}), 400

    try:
        # Get analyzer (lazy initialization)
        analyzer = get_analyzer()
        # Check if we have a valid database to query
        chroma_path = os.path.join(project_root, "chroma")
        if not os.path.exists(chroma_path):
            if not latest_results:
                logger.warning("No analysis has been performed yet")
                return jsonify({'error': 'No analysis has been performed yet. Please analyze a video first.'}), 400
            else:
                video_id = latest_results.get('video_id')
                logger.warning(f"Database not found for video {video_id}")
                return jsonify({'error': f'Database not found for video {video_id}. Please re-analyze the video.'}), 400

        # Convert k to integer if it's provided
        if k is not None:
            try:
                k = int(k)
            except ValueError:
                logger.warning(f"Invalid k parameter: {k}")
                return jsonify({'error': 'Parameter k must be an integer'}), 400

        # Get video ID from latest results or current video ID
        video_id = None
        if latest_results:
            video_id = latest_results.get('video_id')
        if not video_id:
            video_id = analyzer.chroma_manager.get_current_video_id()

        logger.info(f"Processing question: {question} (video_id: {video_id}, k: {k})")

        # Use the analyzer's chroma manager to answer the question
        start_time = time.time()
        result = analyzer.chroma_manager.answer_question(question, k=k, video_id=video_id)

        processing_time = time.time() - start_time
        logger.info(f"Question answered in {processing_time:.2f} seconds")

        # The result already includes the answer, k_used, and other metadata
        return jsonify(result)
    except Exception as e:
        error_msg = f"Error in Q&A: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({'error': error_msg}), 500


# Add a new endpoint to get database status
@app.route('/api/status', methods=['GET'])
def get_status():
    global analyzer
    try:
        chroma_path = os.path.join(project_root, "chroma")
        current_video_id = CURRENT_VIDEO_ID
        
        # Try to get current video ID from analyzer if available
        analyzer_initialized = analyzer is not None
        if analyzer_initialized and analyzer.chroma_manager:
            current_video_id = analyzer.chroma_manager.get_current_video_id() or CURRENT_VIDEO_ID
        
        status = {
            'database_exists': os.path.exists(chroma_path),
            'current_video_id': current_video_id,
            'analyzer_initialized': analyzer_initialized
        }

        # Add metadata from JSON file if it exists (per video)
        if current_video_id:
            metadata_path = os.path.join(chroma_path, current_video_id, "video_metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    status['metadata'] = metadata
                except Exception as e:
                    logger.warning(f"Error reading metadata: {e}")
                    status['metadata_error'] = str(e)

        return jsonify(status)
    except Exception as e:
        logger.error(f"Error in status endpoint: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# Serve static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)


# Serve files from root directory
@app.route('/<path:path>')
def serve_root_files(path):
    file_path = os.path.join(project_root, path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return send_from_directory(project_root, path)
    else:
        return "File not found", 404


if __name__ == '__main__':
    # Optimized Flask development server with faster reloading
    import sys
    
    # Use reloader only in development, exclude unnecessary files
    use_reloader = os.getenv('FLASK_ENV', 'development') == 'development'
    
    app.run(
        debug=True,
        port=5000,
        use_reloader=use_reloader,
        use_debugger=True,
        threaded=True,  # Enable threading for better performance
        extra_files=None  # Can add specific files to watch if needed
    )
