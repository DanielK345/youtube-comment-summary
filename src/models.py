"""
Model-related functionality for YouTube comment analysis.

This module handles:
- HuggingFace sentiment analysis models
- Ollama embeddings and language models
- Sentiment analysis operations
- In-memory vector stores for sentiment summaries
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document


# HuggingFace model configuration
MODEL_NAME = "AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual"
LABEL_MAPPING = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Initialize models (lazy loading can be added if needed)
_tokenizer = None
_hf_model = None


def get_tokenizer():
    """Get or initialize the HuggingFace tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def get_hf_model():
    """Get or initialize the HuggingFace sentiment model."""
    global _hf_model
    if _hf_model is None:
        _hf_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return _hf_model


def get_embedding_function():
    """Returns the embedding function for vector database."""
    return OllamaEmbeddings(model="mxbai-embed-large")


def get_language_model():
    """Returns the Ollama language model."""
    return OllamaLLM(model="llama3.2")


class SentimentAnalyzer:
    """Handles sentiment analysis using HuggingFace models."""
    
    def __init__(self):
        self.tokenizer = get_tokenizer()
        self.model = get_hf_model()
    
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

        for i in range(0, len(comments), batch_size):
            batch = comments[i:i + batch_size]

            try:
                # Tokenize the batch
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )

                # Run model inference
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Get predictions
                predictions = torch.argmax(outputs.logits, dim=1)

                # Categorize comments
                for comment, label in zip(batch, predictions):
                    label = label.item()
                    if label == 2:
                        positives.append(comment)
                    elif label == 1:
                        neutrals.append(comment)
                    else:
                        negatives.append(comment)

            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {str(e)}")
                # Fall back to neutral for this batch if there's an error
                neutrals.extend(batch)

        return [len(neutrals), len(positives), len(negatives)], positives, negatives, neutrals


class SentimentVectorStore:
    """Manages in-memory vector stores for positive and negative sentiment summaries."""
    
    def __init__(self):
        self.embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
        self.positive_db = InMemoryVectorStore(self.embedding_model)
        self.negative_db = InMemoryVectorStore(self.embedding_model)
        self.language_model = OllamaLLM(model="llama3.2")
    
    def reset(self):
        """Reset the in-memory vector stores."""
        print("Resetting sentiment analysis databases...")
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
        self.positive_db.add_documents(self._chunk_documents(documents))
    
    def index_negative_documents(self, documents):
        """Add negative comments to vector database."""
        self.negative_db.add_documents(self._chunk_documents(documents))
    
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
        docs = self.find_related_positive(query)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_POSITIVE)
        chain = prompt | self.language_model
        return chain.invoke({
            "user_query": query,
            "positive_comments": context
        })
    
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
        docs = self.find_related_negative(query)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_NEGATIVE)
        chain = prompt | self.language_model
        return chain.invoke({
            "user_query": query,
            "negative_comments": context
        })
    
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
        # Reset the vector stores before adding new documents
        self.reset()
        
        # Index each into their own vector store
        self.index_positive_documents(positive_comments)
        self.index_negative_documents(negative_comments)
        
        # Summarize each
        pos_summary = self.generate_positive_summary()
        neg_summary = self.generate_negative_summary()
        
        # Save to file
        import os
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("")
            f.write(pos_summary if isinstance(pos_summary, str) else str(pos_summary))
            f.write("\n\n")
            f.write(neg_summary if isinstance(neg_summary, str) else str(neg_summary))
        
        return pos_summary, neg_summary
