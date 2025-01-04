from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain.schema import Document
import google.generativeai as genai
from PIL import Image
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from typing import List, Dict, Optional, Any, Union
from .models import FileContent
import logging
from openai import OpenAI

class EmbeddingHandler:
    """
    Handles all embedding and RAG operations including vector storage and retrieval.
    """

    LANGUAGE_MAP = {
        '.ts': Language.TS,
        '.tsx': Language.TS,
        '.js': Language.JS,
        '.jsx': Language.JS,
        '.py': Language.PYTHON,
    }

    def __init__(self, pinecone_api_key: str, google_api_key: str, 
                 model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 logging_level: int = logging.INFO):
        self.logger = self._setup_logger(logging_level)
        self.model_name = model_name
        self.initialize_clients(pinecone_api_key, google_api_key)
        self.client = OpenAI()

    def _setup_logger(self, logging_level: int) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging_level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def initialize_clients(self, pinecone_api_key: str, google_api_key: str):
        self.pinecone_client = Pinecone(api_key=pinecone_api_key)
        self.pinecone_index = self.pinecone_client.Index("codebase-rag")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        self.sentence_transformer = SentenceTransformer(self.model_name)
        genai.configure(api_key=google_api_key)
        self.gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    def process_file_content(self, file_content: FileContent) -> List[Document]:
        """Process a single file's content into chunks."""
        file_language = self.LANGUAGE_MAP.get(file_content.extension)
        documents = []

        if file_language:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=file_language,
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = splitter.create_documents([file_content.content])
            
            for idx, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "source": file_content.name,
                    "chunk_index": idx,
                    "language": file_language.value
                })
                documents.append(chunk)
        else:
            documents.append(Document(
                page_content=file_content.content,
                metadata={
                    "source": file_content.name,
                    "chunk_index": 0,
                    "language": "unknown"
                }
            ))

        return documents

    def upload_repo_to_pinecone(self, documents: List[Document], repo_url: str):
        """Upload repository content to Pinecone."""
        try:
            self.logger.info(f"Uploading {len(documents)} documents to Pinecone for {repo_url}")
            
            # Créer les embeddings et les métadonnées
            embeddings = []
            metadata = []
            
            for doc in documents:
                embedding = self.sentence_transformer.encode(doc.page_content).tolist()
                embeddings.append(embedding)
                metadata.append({
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                    "language": doc.metadata.get("language", "unknown")
                })

            # Upserter les vecteurs par lots
            batch_size = 100
            for i in range(0, len(embeddings), batch_size):
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadata = metadata[i:i + batch_size]
                
                vectors = [
                    (f"{repo_url}_{j}", emb, meta)
                    for j, (emb, meta) in enumerate(zip(batch_embeddings, batch_metadata))
                ]
                
                self.pinecone_index.upsert(
                    vectors=vectors,
                    namespace=repo_url
                )
                
            self.logger.info("Upload to Pinecone completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error uploading to Pinecone: {str(e)}")
            raise

    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for a given text."""
        return self.sentence_transformer.encode(text).tolist()

    def perform_rag(self, query: str, repo_url: str, selected_model: str = "Google Gemini") -> str:
        """Perform RAG operation."""
        try:
            # Charger le contexte si nécessaire
            if not hasattr(self, '_current_context') or self._current_context is None:
                self._current_context = self.load_repo_context(repo_url)

            # Créer la requête augmentée avec le contexte
            augmented_query = f"""
            Context from codebase:
            {'-' * 80}
            {' '.join(self._current_context[:5])}  # Utiliser les 5 documents les plus pertinents
            {'-' * 80}

            Question: {query}

            Please provide a detailed answer based on the code context above.
            """

            # Obtenir la réponse du modèle
            if selected_model == "Google Gemini":
                response = self.gemini_model.generate_content(augmented_query)
                return response.text
            else:
                raise ValueError(f"Unsupported model: {selected_model}")

        except Exception as e:
            self.logger.error(f"Error in RAG operation: {str(e)}")
            raise

    def _prepare_context(self, matches: Dict) -> str:
        """Prepare context from vector store matches."""
        contexts = [item['metadata']['text'] for item in matches['matches']]
        file_names = [item['metadata']['source'] for item in matches['matches']]
        
        return "\n\n".join(
            [f"File: {file_name}\nContent:\n{context[:500]}..." 
             for file_name, context in zip(file_names, contexts)]
        )

    @staticmethod
    def _create_augmented_query(query: str, context: str) -> str:
        """Create augmented query with context."""
        return f"""
        # Codebase Context:
        {context}

        # Developer Question:
        {query}

        Please provide a response based on the provided context and the specific question.
        """

    def _get_model_response(self, 
                          selected_model: str, 
                          augmented_query: str,
                          image: Optional[str]) -> str:
        """Get response from selected model."""
        try:
            if selected_model == "Google Gemini":
                if image:
                    img = Image.open(image)
                    response = self.gemini_model.generate_content(
                        [augmented_query, img]
                    )
                else:
                    response = self.gemini_model.generate_content(
                        [augmented_query]
                    )
                return response.text
            elif selected_model == "Llama":
                system_prompt = "You are an AI assistant specialized in programming."
                llm_response = self.client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": augmented_query}
                    ]
                )
                return llm_response.choices[0].message.content
            else:
                raise ValueError(f"Unknown model selected: {selected_model}")
        except Exception as e:
            self.logger.error(f"Error getting model response: {str(e)}")
            raise

    def fetch_repos_from_pinecone(self) -> List[str]:
        """Fetch all repository namespaces from Pinecone."""
        try:
            index_stats = self.pinecone_index.describe_index_stats()
            return list(index_stats.get('namespaces', {}).keys())
        except Exception as e:
            self.logger.error(f"Error fetching repos: {str(e)}")
            return []

    def load_repo_context(self, repo_url: str) -> List[str]:
        """Load and prepare the repository context."""
        try:
            # Récupérer les documents depuis Pinecone
            results = self.pinecone_index.query(
                vector=self.sentence_transformer.encode("").tolist(),
                top_k=100,
                include_metadata=True,
                namespace=repo_url
            )
            
            # Extraire le contexte des documents
            context = [
                doc.metadata.get('text', '') 
                for doc in results.matches 
                if doc.metadata and 'text' in doc.metadata
            ]
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error loading repo context: {str(e)}")
            raise