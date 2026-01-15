"""
Model-related functionality for YouTube comment analysis.

This module handles:
- HuggingFace sentiment analysis models
- Google Gemini embeddings and language models
- Sentiment analysis operations
- In-memory vector stores for sentiment summaries
"""

import os
import logging
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from tqdm import tqdm
from src.utils import setup_logger

# Load .env file from project root (parent of src directory)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_file_path = os.path.join(project_root, '.env')
env_loaded = load_dotenv(env_file_path)

# HuggingFace model configuration
MODEL_NAME = "AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual"
LABEL_MAPPING = {0: "Negative", 1: "Neutral", 2: "Positive"}


class ModelManager:
    """Manages all AI models (HuggingFace, Gemini embeddings, Gemini LLM)."""
    
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize ModelManager.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or setup_logger(__name__)
        self._tokenizer = None
        self._hf_model = None
        self._embedding_model = None
        self._language_model = None
        self._initialized = False
    
    def initialize(self):
        """Initialize all models with progress tracking."""
        if self._initialized:
            self.logger.info("Models already initialized")
            return
        
        try:
            self.logger.info("Initializing models...")
            
            # Initialize HuggingFace model
            self._load_huggingface_model()
            
            # Initialize LLM + embeddings
            #
            # Prefer Gemini, but fall back to Ollama (Llama 3.2) when Gemini
            # cannot be initialized (missing key, network/import errors, etc.).
            try:
                self._load_gemini_models()
            except Exception as e:
                self.logger.warning(
                    "Gemini models could not be loaded; falling back to Ollama (Llama 3.2). "
                    f"Reason: {e}"
                )
                self._load_llama_models()
            
            self._initialized = True
            self.logger.info("All models initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}", exc_info=True)
            raise
    
    def _load_huggingface_model(self):
        """Load HuggingFace sentiment analysis model."""
        try:
            self.logger.info(f"Loading HuggingFace model: {MODEL_NAME}")
            with tqdm(total=2, desc="Loading HuggingFace model", unit="step") as pbar:
                self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                pbar.update(1)
                self._hf_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
                pbar.update(1)
            self.logger.info("HuggingFace model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load HuggingFace model: {e}", exc_info=True)
            raise
    
    def _load_gemini_models(self):
        """Load Google Gemini embedding and language models."""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                env_file_path = os.path.join(project_root, '.env')
                # Raise to trigger Ollama fallback in initialize()
                raise ValueError(
                    "GOOGLE_API_KEY environment variable is required to use Gemini.\n"
                    f"Create a .env file at {env_file_path} with: GOOGLE_API_KEY=your-api-key\n"
                    f"Note: .env file {'exists' if os.path.exists(env_file_path) else 'NOT FOUND'} at {env_file_path}"
                )
            
            # NOTE:
            # `gemini-pro` was used by older SDKs, but many installs now require
            # the newer model IDs (e.g. `gemini-1.5-flash`, `gemini-1.5-pro`).
            # Make the model configurable and default to a widely available one.
            gemini_chat_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

            self.logger.info("Loading Google Gemini models...")
            with tqdm(total=2, desc="Loading Gemini models", unit="model") as pbar:
                self._embedding_model = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=api_key
                )
                pbar.update(1)
                self._language_model = ChatGoogleGenerativeAI(
                    model=gemini_chat_model,
                    google_api_key=api_key,
                    temperature=0.7
                )
                pbar.update(1)
            self.logger.info("Google Gemini models loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Gemini models: {e}", exc_info=True)
            raise

    def _load_llama_models(self):
        """
        Load Ollama-backed models as a fallback.

        - Chat model: defaults to `llama3.2` (configurable via OLLAMA_CHAT_MODEL)
        - Embeddings: defaults to `mxbai-embed-large` (configurable via OLLAMA_EMBED_MODEL)
        - Base URL: defaults to `http://localhost:11434` (configurable via OLLAMA_BASE_URL)

        Requires Ollama running locally.
        """
        try:
            # Lazy import so Gemini-only users don't need Ollama installed/running.
            from langchain_ollama import ChatOllama
            from langchain_ollama import EmbedOllama

            chat_model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
            embed_model = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

            self.logger.info(
                f"Loading Ollama models (chat={chat_model}, embed={embed_model}, base_url={base_url})..."
            )
            with tqdm(total=2, desc="Loading Ollama models", unit="model") as pbar:
                self._embedding_model = EmbedOllama(model=embed_model, base_url=base_url)
                pbar.update(1)
                self._language_model = ChatOllama(model=chat_model, base_url=base_url, temperature=0.7)
                pbar.update(1)  

            self.logger.info("Ollama models loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Ollama models: {e}", exc_info=True)
            raise
    
    def get_embedding_function(self):
        """Get the embedding function for vector database."""
        if not self._initialized:
            self.initialize()
        if self._embedding_model is None:
            raise RuntimeError("Embedding model not initialized")
        return self._embedding_model
    
    def get_language_model(self):
        """Get the language model."""
        if not self._initialized:
            self.initialize()
        if self._language_model is None:
            raise RuntimeError("Language model not initialized")
        return self._language_model
    
    def get_tokenizer(self):
        """Get the HuggingFace tokenizer."""
        if not self._initialized:
            self.initialize()
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        return self._tokenizer
    
    def get_hf_model(self):
        """Get the HuggingFace sentiment model."""
        if not self._initialized:
            self.initialize()
        if self._hf_model is None:
            raise RuntimeError("HuggingFace model not initialized")
        return self._hf_model


class SentimentAnalyzer:
    """Handles sentiment analysis using HuggingFace models."""
    
    def __init__(self, model_manager: ModelManager = None, logger: logging.Logger = None):
        """
        Initialize SentimentAnalyzer.
        
        Args:
            model_manager: ModelManager instance (creates new one if not provided)
            logger: Optional logger instance
        """
        self.logger = logger or setup_logger(__name__)
        self.model_manager = model_manager or ModelManager(logger=self.logger)
        self.model_manager.initialize()
        self.tokenizer = self.model_manager.get_tokenizer()
        self.model = self.model_manager.get_hf_model()
    
    def analyze_sentiment(self, comments):
        """
        Analyze sentiment of comments using HuggingFace model with batch processing.
        
        Args:
            comments: List of comment strings to analyze
            
        Returns:
            Tuple of (sentiment_counts, positive_comments, negative_comments, neutral_comments)
            where sentiment_counts is [neutral_count, positive_count, negative_count]
        """
        batch_size = 100
        positives, neutrals, negatives = [], [], []
        
        self.logger.info(f"Analyzing sentiment for {len(comments)} comments")

        with tqdm(total=len(comments), desc="Analyzing sentiment", unit="comment") as pbar:
            for i in range(0, len(comments), batch_size):
                batch = comments[i:i + batch_size]

                try:
                    # Lazy import torch for faster module loading
                    from torch import argmax
                    from torch.autograd import no_grad
                    
                    # Tokenize the batch
                    inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    )

                    # Run model inference
                    with no_grad():
                        outputs = self.model(**inputs)

                    # Get predictions
                    predictions = argmax(outputs.logits, dim=1)

                    # Categorize comments
                    for comment, label in zip(batch, predictions):
                        label = label.item()
                        if label == 2:
                            positives.append(comment)
                        elif label == 1:
                            neutrals.append(comment)
                        else:
                            negatives.append(comment)
                    
                    pbar.update(len(batch))

                except Exception as e:
                    self.logger.error(f"Error processing batch {i//batch_size}: {str(e)}", exc_info=True)
                    # Fall back to neutral for this batch if there's an error
                    neutrals.extend(batch)
                    pbar.update(len(batch))

        self.logger.info(f"Sentiment analysis complete: {len(positives)} positive, {len(negatives)} negative, {len(neutrals)} neutral")
        return [len(neutrals), len(positives), len(negatives)], positives, negatives, neutrals


class SentimentVectorStore:
    """Manages in-memory vector stores for positive and negative sentiment summaries."""
    
    def __init__(self, model_manager: ModelManager = None, logger: logging.Logger = None):
        """
        Initialize SentimentVectorStore.
        
        Args:
            model_manager: ModelManager instance (creates new one if not provided)
            logger: Optional logger instance
        """
        self.logger = logger or setup_logger(__name__)
        self.model_manager = model_manager or ModelManager(logger=self.logger)
        self.model_manager.initialize()
        
        try:
            self.embedding_model = self.model_manager.get_embedding_function()
            self.positive_db = InMemoryVectorStore(self.embedding_model)
            self.negative_db = InMemoryVectorStore(self.embedding_model)
            self.language_model = self.model_manager.get_language_model()
            self.logger.info("SentimentVectorStore initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize SentimentVectorStore: {e}", exc_info=True)
            raise
    
    def reset(self):
        """Reset the in-memory vector stores."""
        self.logger.info("Resetting sentiment analysis databases...")
        self.positive_db = InMemoryVectorStore(self.embedding_model)
        self.negative_db = InMemoryVectorStore(self.embedding_model)
    
    def _chunk_documents(self, raw_documents):
        """Split documents into chunks for better vector search."""
        document_objects = [Document(page_content=doc) for doc in raw_documents]
        text_processor = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=10,
            add_start_index=True
        )
        return text_processor.split_documents(document_objects)
    
    def index_positive_documents(self, documents):
        """Add positive comments to vector database."""
        try:
            chunks = self._chunk_documents(documents)
            self.positive_db.add_documents(chunks)
            self.logger.debug(f"Indexed {len(documents)} positive comments")
        except Exception as e:
            self.logger.error(f"Error indexing positive documents: {e}", exc_info=True)
            raise
    
    def index_negative_documents(self, documents):
        """Add negative comments to vector database."""
        try:
            chunks = self._chunk_documents(documents)
            self.negative_db.add_documents(chunks)
            self.logger.debug(f"Indexed {len(documents)} negative comments")
        except Exception as e:
            self.logger.error(f"Error indexing negative documents: {e}", exc_info=True)
            raise
    
    def find_related_positive(self, query, k=200):
        """Find related positive comments for a query."""
        return self.positive_db.similarity_search(query, k=k)
    
    def find_related_negative(self, query, k=200):
        """Find related negative comments for a query."""
        return self.negative_db.similarity_search(query, k=k)
    
    def generate_positive_summary(self, query="Summarize main point of these comments."):
        """Generate a summary of positive comments."""
        PROMPT_TEMPLATE_POSITIVE = """
        You are a YouTube sentiment analysis assistant. Your task is to summarize YouTube video comments.
        Below are the positive comments:
        ---
        {positive_comments}
        ---
        Please summarize the main points expressed in these positive comments.
        Return only a bullet-point list of the main takeaways with the layout of 1 line break for each point.
        Start the summary with bullet points right away, and do not include any other text.
        Note that just include 0-3 main points.
        Do not include any negative comments or neutral comments in the summary if they are present.
        """
        try:
            docs = self.find_related_positive(query)
            context = "\n".join([doc.page_content for doc in docs])
            prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_POSITIVE)
            chain = prompt | self.language_model
            result = chain.invoke({
                "user_query": query,
                "positive_comments": context
            })
            # ChatGoogleGenerativeAI returns AIMessage, extract content if needed
            if hasattr(result, 'content'):
                return result.content
            return result
        except Exception as e:
            # If Gemini fails at runtime (e.g., model NOT_FOUND), fall back to Ollama once.
            self.logger.error(f"Error generating positive summary: {e}", exc_info=True)
            if "NOT_FOUND" in str(e) or "models/" in str(e):
                self.logger.warning("Gemini call failed; switching SentimentVectorStore to Ollama fallback and retrying once.")
                self.model_manager._load_llama_models()
                self.embedding_model = self.model_manager.get_embedding_function()
                self.language_model = self.model_manager.get_language_model()
                self.reset()
                self.index_positive_documents([doc.page_content for doc in docs])
                chain = prompt | self.language_model
                result = chain.invoke({
                    "user_query": query,
                    "positive_comments": context
                })
                if hasattr(result, 'content'):
                    return result.content
                return result
            raise
    
    def generate_negative_summary(self, query="Summarize main point of the negative comments."):
        """Generate a summary of negative comments."""
        PROMPT_TEMPLATE_NEGATIVE = """
        You are a YouTube sentiment analysis assistant. Your task is to analyze and summarize YouTube video comments.
        Below are the negative comments:
        ---
        {negative_comments}
        ---
        Please summarize the main points expressed in these negative comments
        Return only a bullet-point list of the main takeaways with the layout of 1 line break for each point.
        Start the summary with bullet points right away, and do not include any other text.
        Note that just include 0-3 main points related to the negative comments.
        Do not include any the positive comments or neutral comments if they are present.
        """
        try:
            docs = self.find_related_negative(query)
            context = "\n".join([doc.page_content for doc in docs])
            prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_NEGATIVE)
            chain = prompt | self.language_model
            result = chain.invoke({
                "user_query": query,
                "negative_comments": context
            })
            # ChatGoogleGenerativeAI returns AIMessage, extract content if needed
            if hasattr(result, 'content'):
                return result.content
            return result
        except Exception as e:
            self.logger.error(f"Error generating negative summary: {e}", exc_info=True)
            if "NOT_FOUND" in str(e) or "models/" in str(e):
                self.logger.warning("Gemini call failed; switching SentimentVectorStore to Ollama fallback and retrying once.")
                self.model_manager._load_llama_models()
                self.embedding_model = self.model_manager.get_embedding_function()
                self.language_model = self.model_manager.get_language_model()
                self.reset()
                self.index_negative_documents([doc.page_content for doc in docs])
                chain = prompt | self.language_model
                result = chain.invoke({
                    "user_query": query,
                    "negative_comments": context
                })
                if hasattr(result, 'content'):
                    return result.content
                return result
            raise
    
    def summarize_both_sentiments(self, positive_comments, negative_comments, output_file="sentiment_summary.txt"):
        """
        Summarize both positive and negative sentiments.
        
        Args:
            positive_comments: List of positive comment strings
            negative_comments: List of negative comment strings
            output_file: Path to save the summary
            
        Returns:
            Tuple of (positive_summary, negative_summary)
        """
        try:
            # Reset the vector stores before adding new documents
            self.reset()
            
            # Index each into their own vector store
            self.index_positive_documents(positive_comments)
            self.index_negative_documents(negative_comments)
            
            # Summarize each
            self.logger.info("Generating positive summary...")
            pos_summary = self.generate_positive_summary()
            
            self.logger.info("Generating negative summary...")
            neg_summary = self.generate_negative_summary()
            
            # Save to file
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("")
                f.write(pos_summary if isinstance(pos_summary, str) else str(pos_summary))
                f.write("\n\n")
                f.write(neg_summary if isinstance(neg_summary, str) else str(neg_summary))
            
            self.logger.info(f"Sentiment summaries saved to {output_file}")
            return pos_summary, neg_summary
        except Exception as e:
            self.logger.error(f"Error summarizing sentiments: {e}", exc_info=True)
            raise


# Backward compatibility functions
def get_embedding_function():
    """Backward compatibility wrapper."""
    manager = ModelManager()
    manager.initialize()
    return manager.get_embedding_function()


def get_language_model():
    """Backward compatibility wrapper."""
    manager = ModelManager()
    manager.initialize()
    return manager.get_language_model()


def get_tokenizer():
    """Backward compatibility wrapper."""
    manager = ModelManager()
    manager.initialize()
    return manager.get_tokenizer()


def get_hf_model():
    """Backward compatibility wrapper."""
    manager = ModelManager()
    manager.initialize()
    return manager.get_hf_model()
