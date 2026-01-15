"""
ChromaDB operations for storing and retrieving YouTube comments.

This module handles:
- ChromaDB connection and management
- Saving comments to vector database
- Reading comments from vector database
- Video-specific database management
"""

import os
import time
import json
import gc
import logging
from tqdm import tqdm

from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.models import ModelManager
from src.utils import setup_logger


class ChromaDBManager:
    """Manages ChromaDB operations for YouTube comments."""
    
    def __init__(self, chroma_path: str = "chroma", model_manager: ModelManager = None, logger: logging.Logger = None):
        """
        Initialize ChromaDBManager.
        
        Args:
            chroma_path: Base path for ChromaDB storage
            model_manager: ModelManager instance (creates new one if not provided)
            logger: Optional logger instance
        """
        self.chroma_path = chroma_path
        self.current_video_id = None
        self.logger = logger or setup_logger(__name__)
        self.model_manager = model_manager or ModelManager(logger=self.logger)
        
        # Ensure base directory exists
        os.makedirs(self.chroma_path, exist_ok=True)
        self.logger.info(f"ChromaDBManager initialized with path: {self.chroma_path}")
    
    def get_chroma_db(self, video_id: str):
        """
        Create and return a Chroma database connection with proper error handling.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Chroma database instance
        """
        try:
            # Set up directory path for this specific video
            video_specific_path = os.path.join(self.chroma_path, video_id)
            os.makedirs(video_specific_path, exist_ok=True)

            # Initialize model manager if needed
            self.model_manager.initialize()
            embedding_function = self.model_manager.get_embedding_function()

            # Create a client without persistence first to avoid tenant errors
            db = Chroma(
                collection_name=f"comments_{video_id}",
                embedding_function=embedding_function,
                persist_directory=video_specific_path
            )

            # Update the current video ID
            self.current_video_id = video_id

            self.logger.info(f"Successfully connected to Chroma database for video ID: {video_id}")
            return db

        except Exception as e:
            self.logger.error(f"Error connecting to Chroma database: {e}", exc_info=True)
            # Create an alternative path with timestamp if there's an issue
            timestamp = int(time.time())
            alt_path = os.path.join(self.chroma_path, f"{video_id}_{timestamp}")
            os.makedirs(alt_path, exist_ok=True)

            self.logger.warning(f"Attempting to create alternative database at {alt_path}")

            # Try with the alternative path
            try:
                embedding_function = self.model_manager.get_embedding_function()
                db = Chroma(
                    collection_name=f"comments_{video_id}_{timestamp}",
                    embedding_function=embedding_function,
                    persist_directory=alt_path
                )
                return db
            except Exception as e2:
                self.logger.error(f"Failed to create alternative database: {e2}", exc_info=True)
                raise RuntimeError(f"Cannot initialize Chroma database: {e2}")
    
    def close_connection(self):
        """Close any open connections to the Chroma database."""
        self.logger.info("Closing ChromaDB connection...")
        # Force garbage collection to release file handles
        gc.collect()
        time.sleep(1)  # Give a moment for resources to be released
        self.logger.info("ChromaDB connection closed")
    
    def save_comments_to_chroma(self, comments: list, video_id: str):
        """
        Populate comments into Chroma database, clearing previous data if video ID changed.

        Args:
            comments: List of comment dictionaries with 'author', 'comment', 'likes', etc.
            video_id: YouTube video ID to check if we need to refresh the database

        Returns:
            Number of comments saved to the database
        """
        # Check if we already have a database for this video
        if self.current_video_id == video_id and os.path.exists(self.chroma_path):
            video_path = os.path.join(self.chroma_path, video_id)
            if os.path.exists(video_path):
                self.logger.info(f"Using existing Chroma database for video ID: {video_id}")
                return len(comments)

        # If video ID changed or no database exists, rebuild it
        if os.path.exists(self.chroma_path):
            self.logger.info(f"Video ID changed from {self.current_video_id} to {video_id}. Creating new database.")

            # Close any open connections before creating new one
            self.close_connection()

            try:
                # Create a new subfolder for this video
                video_path = os.path.join(self.chroma_path, video_id)
                os.makedirs(video_path, exist_ok=True)
            except Exception as e:
                self.logger.error(f"Error handling Chroma directory: {e}", exc_info=True)

        # Get the Chroma database
        db = self.get_chroma_db(video_id)

        # Create Document objects for each comment
        self.logger.info(f"Creating documents for {len(comments)} comments...")
        documents = []
        
        with tqdm(total=len(comments), desc="Preparing documents", unit="comment") as pbar:
            for idx, comment in enumerate(comments, start=1):
                # Format the comment text to include author and likes
                if comment.get('likes', 0) > 0:
                    content = f"{comment['author']} [👍 {comment['likes']}]:\n{comment['comment']}"
                else:
                    content = f"{comment['author']}:\n{comment['comment']}"

                # Add metadata
                metadata = {
                    "source": f"Comment {idx}",
                    "author": comment['author'],
                    "likes": comment.get('likes', 0)
                }

                if 'replied_to' in comment:
                    # Add 'replied_to' for replies
                    metadata['replied_to'] = comment['replied_to']
                    # Mark as reply in the content for better context
                    content = f"[REPLY to {comment['replied_to']}] {content}"

                document = Document(page_content=content, metadata=metadata)
                documents.append(document)
                pbar.update(1)

        # Add documents to Chroma in batches to avoid memory issues
        batch_size = 100
        self.logger.info(f"Adding {len(documents)} documents to ChromaDB in batches of {batch_size}...")
        
        with tqdm(total=len(documents), desc="Saving to ChromaDB", unit="comment") as pbar:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                db.add_documents(batch)
                pbar.update(len(batch))
                self.logger.debug(f"Added batch of {len(batch)} comments to Chroma (total {i + len(batch)})")

        # Save video metadata to help with QA
        metadata_path = os.path.join(self.chroma_path, video_id, "video_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump({
                "video_id": video_id,
                "comment_count": len(documents)
            }, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Successfully added all {len(documents)} comments to Chroma database.")
        return len(documents)
    
    def read_comments_from_chroma(self, video_id: str = None):
        """
        Read comments from the Chroma database.
        
        Args:
            video_id: Specific video ID to read from (defaults to current)
            
        Returns:
            List of comment strings
        """
        # Use current video ID if none specified
        if video_id is None:
            video_id = self.current_video_id

        if video_id is None:
            raise ValueError("No video ID specified and no current video ID set")

        # Connect to the existing Chroma database
        try:
            self.logger.info(f"Reading comments from ChromaDB for video ID: {video_id}")
            video_specific_path = os.path.join(self.chroma_path, video_id)
            
            self.model_manager.initialize()
            embedding_function = self.model_manager.get_embedding_function()
            
            db = Chroma(
                collection_name=f"comments_{video_id}",
                embedding_function=embedding_function,
                persist_directory=video_specific_path
            )

            # Get all documents from the database
            results = db.get()

            # Extract comments from the documents
            comments = []
            with tqdm(total=len(results.get('documents', [])), desc="Reading comments", unit="comment") as pbar:
                for doc in results.get('documents', []):
                    # Each document has format "Author [👍 Likes]:\nComment" or "Author:\nComment"
                    # Split to get just the comment part
                    parts = doc.split('\n', 1)
                    if len(parts) > 1:
                        # Just the comment text, not the author
                        comments.append(parts[1])
                    pbar.update(1)

            self.logger.info(f"Read {len(comments)} comments from ChromaDB")
            return comments

        except Exception as e:
            self.logger.error(f"Error reading from Chroma database: {e}", exc_info=True)
            return []
    
    def calculate_optimal_k(self, total_comments: int):
        """
        Calculate the optimal k value based on total comment count.
        Optimized based on testing that k=50-70 works best for ~200 comments.

        Args:
            total_comments: Total number of comments in the database

        Returns:
            Recommended k value
        """
        # For very small comment sets (<50), use a higher percentage (60-70%)
        if total_comments < 50:
            return max(10, min(int(total_comments * 0.7), total_comments))

        # For small comment sets (50-200), scale between 40-30% of total
        elif total_comments < 200:
            # Linear scaling from 40% at 50 comments to 30% at 200 comments
            percent = 0.4 - ((total_comments - 50) / 150) * 0.1
            return max(20, min(int(total_comments * percent), total_comments))

        # For medium comment sets (200-1000), scale between 30-20% of total
        elif total_comments < 1000:
            # Linear scaling from 30% at 200 comments to 20% at 1000 comments
            percent = 0.3 - ((total_comments - 200) / 800) * 0.1
            return max(60, min(int(total_comments * percent), total_comments))

        # For large comment sets (1000-5000), scale between 20-10% of total
        elif total_comments < 5000:
            # Linear scaling from 20% at 1000 comments to 10% at 5000 comments
            percent = 0.2 - ((total_comments - 1000) / 4000) * 0.1
            return max(200, min(int(total_comments * percent), 500))

        # For very large comment sets (>5000), use 10% with a cap at 600
        else:
            return min(int(total_comments * 0.1), 600)
    
    def answer_question(self, question: str, k: int = None, video_id: str = None):
        """
        Answer a question based on the YouTube comments data with improved analysis.

        Args:
            question: The user's question about the video comments
            k: Number of relevant comments to retrieve for context (auto-calculated if None)
            video_id: Specific video ID to use (defaults to current)

        Returns:
            Dictionary with answer and metadata
        """
        from langchain_core.prompts import ChatPromptTemplate
        
        # Start timing
        start_time = time.time()
        
        if video_id is None:
            video_id = self.current_video_id

        if video_id is None:
            raise ValueError("No video ID specified and no current video ID set")

        # Connect to the Chroma vector store for this specific video
        video_specific_path = os.path.join(self.chroma_path, video_id)
        
        self.model_manager.initialize()
        embedding_function = self.model_manager.get_embedding_function()
        
        db = Chroma(
            collection_name=f"comments_{video_id}",
            embedding_function=embedding_function,
            persist_directory=video_specific_path
        )

        # Get the total number of documents in the database
        try:
            doc_count = len(db.get()['ids'])
        except:
            self.logger.warning("Could not get document count, defaulting to 0")
            doc_count = 0

        # Calculate optimal k if not specified
        if k is None:
            k = self.calculate_optimal_k(doc_count)
            self.logger.info(f"Auto-calculated optimal k value: {k} (based on {doc_count} total comments)")

        # Store the original k value for reporting
        k_used = k

        # Adjust k if it's larger than the number of available documents
        if k > doc_count:
            self.logger.warning(f"Adjusting k from {k} to {doc_count} (total available documents)")
            k = doc_count

        if k == 0:
            return {
                'answer': "There are no comments available to analyze. Please check if the video exists and has public comments.",
                'k_used': 0,
                'comments_total': 0,
                'processing_time': "0.00 seconds"
            }

        # Improved prompt template with better structure and instructions
        PROMPT_TEMPLATE = """
        You are a YouTube comment analyst answering questions about video comments.

        QUESTION: {question}

        Below are relevant comments from the video:
        {context}

        Answer the question ONLY using information in these comments. Your response should:

        1. Start with a direct answer addressing the question
        2. Group similar opinions together
        3. Include specific quotes from commenters as evidence when relevant
        4. Stay STRICTLY focused on the question

        For comparison or preference questions:
        - Use clear headings
        - Use bullet points for listing multiple points
        - Structure information logically by categories

        For numerical questions (counts, percentages, etc.):
        - Provide a direct numerical answer if possible
        - Explain how you arrived at this number
        - Include specific evidence from comments

        DO NOT invent information not present in the comments.
        DO NOT include follow-up questions or recommendations unless requested.
        FOCUS only on answering exactly what was asked: {question}
        """

        self.logger.info(f"Retrieving {k} most relevant comments for the question...")

        # Retrieve relevant documents
        try:
            results = db.similarity_search_with_score(question, k=k)
        except Exception as e:
            self.logger.error(f"Error during similarity search: {e}", exc_info=True)
            return {
                'answer': "An error occurred while retrieving comments. Please try again.",
                'k_used': k_used,
                'comments_total': doc_count,
                'processing_time': f"{time.time() - start_time:.2f} seconds",
                'error': str(e)
            }

        retrieval_time = time.time() - start_time
        self.logger.info(f"Retrieved {len(results)} comments in {retrieval_time:.2f} seconds")

        # Sort comments by relevance score to prioritize most relevant ones
        sorted_results = sorted(results, key=lambda x: x[1])

        # Take only the most relevant comments to avoid overwhelming the LLM
        top_results = sorted_results[:min(k, len(sorted_results))]

        # Build context string from retrieved documents with comment numbering
        context_parts = []
        for i, (doc, score) in enumerate(top_results):
            context_parts.append(f"[{i + 1}] {doc.page_content}")

        context_text = "\n\n".join(context_parts)

        # Format prompt with context
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(question=question, context=context_text)

        # Generating answer with language model
        self.logger.info("Generating answer with language model...")
        model = self.model_manager.get_language_model()

        generation_start = time.time()
        try:
            response_text = model.invoke(prompt)
        except Exception as e:
            # If Gemini fails at runtime (common: NOT_FOUND for a model name / API version),
            # fall back to Ollama once.
            if "NOT_FOUND" in str(e) or "models/" in str(e):
                self.logger.warning(
                    "Language model call failed; falling back to Ollama (Llama 3.2) and retrying once. "
                    f"Reason: {e}"
                )
                self.model_manager._load_llama_models()
                model = self.model_manager.get_language_model()
                response_text = model.invoke(prompt)
            else:
                raise
        # ChatGoogleGenerativeAI returns AIMessage, extract content if needed
        if hasattr(response_text, 'content'):
            response_text = response_text.content
        generation_time = time.time() - generation_start

        total_time = time.time() - start_time
        self.logger.info(f"Answer generated in {generation_time:.2f} seconds")
        self.logger.info(f"Total processing time: {total_time:.2f} seconds")

        # Return both the answer and metadata
        return {
            'answer': response_text,
            'k_used': k_used,
            'comments_total': doc_count,
            'processing_time': f"{total_time:.2f} seconds"
        }
    
    def generate_comment_summary(self, video_id: str = None):
        """
        Generate a general summary of all comments with improved diversity.
        
        Args:
            video_id: Specific video ID to summarize (defaults to current)
            
        Returns:
            Summary text string
        """
        import random
        from langchain_core.prompts import ChatPromptTemplate
        
        # Use current video ID if none provided
        if video_id is None:
            video_id = self.current_video_id

        if video_id is None:
            raise ValueError("No video ID specified and no current video ID set")

        # Connect to the Chroma vector store for this specific video
        try:
            video_specific_path = os.path.join(self.chroma_path, video_id)
            
            self.model_manager.initialize()
            embedding_function = self.model_manager.get_embedding_function()
            
            db = Chroma(
                collection_name=f"comments_{video_id}",
                embedding_function=embedding_function,
                persist_directory=video_specific_path
            )

            # Get the total number of documents in the database
            try:
                doc_count = len(db.get()['ids'])
            except:
                self.logger.warning("Could not get document count, defaulting to 0")
                doc_count = 0

            if doc_count == 0:
                self.logger.warning("No comments found to summarize")
                return "No comments available to summarize."

            # Calculate appropriate k value based on document count
            # For summaries, we want a larger sample than for QA but not too large
            base_k = self.calculate_optimal_k(doc_count)
            # Double the QA k, but don't exceed doc count
            k = min(base_k * 2, doc_count)

            self.logger.info(f"Using k={k} for summary (based on {doc_count} total comments)")

            # Use a more balanced prompt that emphasizes diversity
            PROMPT_TEMPLATE = """
            You are a YouTube comment summarizer. Below is a collection of user comments extracted from a video.

            {context}

            ---

            Please write a summary highlighting the key points and general sentiment expressed in these comments.
            Focus on providing a well-rounded overview in less than 5 paragraphs.

            IMPORTANT: Make sure to cover diverse topics from the comments. Do not focus too much on any single
            topic or theme, even if many comments discuss it. Instead, try to capture the overall breadth of
            topics and opinions present across ALL comments.
            """

            # Get a mix of targeted and random comments for better diversity
            similarity_k = k // 2
            random_k = k - similarity_k

            self.logger.info(f"Retrieving {similarity_k} targeted comments and {random_k} random comments for summary...")

            # Get targeted comments using similarity search
            try:
                results1 = db.similarity_search_with_score(
                    "summarize youtube comments", k=similarity_k)
            except Exception as e:
                self.logger.error(f"Error during similarity search: {e}", exc_info=True)
                # Fall back to getting all documents
                results1 = []

            # Get random comments for diversity
            try:
                all_docs = db.get()
                random_indices = random.sample(range(doc_count), min(random_k, doc_count))
                random_docs = [
                    (Document(page_content=all_docs['documents'][i]), 1.0) for i in random_indices]
            except Exception as e:
                self.logger.error(f"Error getting random documents: {e}", exc_info=True)
                random_docs = []

            # Combine both sets
            combined_results = results1 + random_docs

            if not combined_results:
                self.logger.error("No comments retrieved for summary")
                return "Unable to generate summary due to data retrieval issues."

            # Build context string from retrieved documents
            context_text = "\n\n---\n\n".join(
                [doc.page_content for doc, _score in combined_results])

            # Format prompt with context
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text)

            # Use Google Gemini model
            self.logger.info("Generating summary with language model...")
            model = self.model_manager.get_language_model()
            try:
                response_text = model.invoke(prompt)
            except Exception as e:
                if "NOT_FOUND" in str(e) or "models/" in str(e):
                    self.logger.warning(
                        "Language model call failed; falling back to Ollama (Llama 3.2) and retrying once. "
                        f"Reason: {e}"
                    )
                    self.model_manager._load_llama_models()
                    model = self.model_manager.get_language_model()
                    response_text = model.invoke(prompt)
                else:
                    raise
            # ChatGoogleGenerativeAI returns AIMessage, extract content if needed
            if hasattr(response_text, 'content'):
                response_text = response_text.content

            # Save the output to a file
            output_dir = os.path.join(self.chroma_path, video_id)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "overall_summary.txt"), "w", encoding="utf-8") as f:
                f.write(response_text)

            self.logger.info(f"Overall summary saved to {output_dir}/overall_summary.txt")
            return response_text

        except Exception as e:
            self.logger.error(f"Error generating comment summary: {e}", exc_info=True)
            return f"Error generating comment summary: {str(e)}"
    
    def get_current_video_id(self):
        """Get the current video ID being analyzed."""
        return self.current_video_id
    
    def get_chroma_path(self):
        """Get the Chroma database path."""
        return self.chroma_path


# Global instance for backward compatibility
_chroma_manager = None
CHROMA_PATH = "chroma"
CURRENT_VIDEO_ID = None


def _get_global_manager():
    """Get or create global ChromaDBManager instance."""
    global _chroma_manager
    if _chroma_manager is None:
        _chroma_manager = ChromaDBManager(chroma_path=CHROMA_PATH)
    return _chroma_manager


# Backward compatibility functions
def close_chroma_connection():
    """Backward compatibility wrapper."""
    _get_global_manager().close_connection()


def get_chroma_db(video_id):
    """Backward compatibility wrapper."""
    manager = _get_global_manager()
    db = manager.get_chroma_db(video_id)
    global CURRENT_VIDEO_ID
    CURRENT_VIDEO_ID = manager.current_video_id
    return db


def save_comments_to_chroma(comments, video_id):
    """Backward compatibility wrapper."""
    manager = _get_global_manager()
    result = manager.save_comments_to_chroma(comments, video_id)
    global CURRENT_VIDEO_ID
    CURRENT_VIDEO_ID = manager.current_video_id
    return result


def read_comments_from_chroma(video_id=None):
    """Backward compatibility wrapper."""
    manager = _get_global_manager()
    if video_id is None:
        video_id = manager.current_video_id
    return manager.read_comments_from_chroma(video_id)


def calculate_optimal_k(total_comments):
    """Backward compatibility wrapper."""
    return _get_global_manager().calculate_optimal_k(total_comments)


def answer_question(question, k=None, video_id=None):
    """Backward compatibility wrapper."""
    manager = _get_global_manager()
    if video_id is None:
        video_id = manager.current_video_id
    return manager.answer_question(question, k=k, video_id=video_id)


def generate_comment_summary(video_id=None):
    """Backward compatibility wrapper."""
    manager = _get_global_manager()
    if video_id is None:
        video_id = manager.current_video_id
    return manager.generate_comment_summary(video_id)


def get_current_video_id():
    """Backward compatibility wrapper."""
    return _get_global_manager().get_current_video_id()


def get_chroma_path():
    """Backward compatibility wrapper."""
    return _get_global_manager().get_chroma_path()


def get_embedding_function():
    """Backward compatibility wrapper - import from models."""
    from src.models import get_embedding_function as _get_embedding_function
    return _get_embedding_function()
