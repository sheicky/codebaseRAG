import streamlit as st
from dotenv import load_dotenv
import os
import asyncio
from typing import Optional
from rag.embedding import EmbeddingHandler
from rag.get_repo import RepositoryHandler
from rag.models import Message
import logging
from datetime import datetime
from github import Github
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class CodebaseRAGApp:
    """
    Main application class for the Codebase RAG Chat interface.
    
    This class manages the Streamlit interface and coordinates between
    the repository and embedding handlers.
    """

    def __init__(self):
        """Initialize the application with configurations and handlers."""
        load_dotenv()
        self.setup_logging()
        self.initialize_handlers()
        self.initialize_session_state()
        self.github = Github(os.getenv("GITHUB_TOKEN"))  # Add GitHub client

    def setup_logging(self):
        """Set up application logging."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def initialize_handlers(self):
        """Initialize repository and embedding handlers."""
        try:
            self.embedding_handler = EmbeddingHandler(
                pinecone_api_key=os.getenv("PINECONE_API_KEY"),
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            self.repo_handler = RepositoryHandler()
        except Exception as e:
            self.logger.error(f"Failed to initialize handlers: {str(e)}")
            st.error("Failed to initialize application. Please check your API keys.")
            st.stop()

    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if "theme" not in st.session_state:
            st.session_state.theme = "dark"
        if "selected_codebase" not in st.session_state:
            st.session_state.selected_codebase = None
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "repos_fetched" not in st.session_state:
            st.session_state.repos_fetched = False
        if "repos" not in st.session_state:
            st.session_state.repos = []
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = "Google Gemini"
        if "processing_status" not in st.session_state:
            st.session_state.processing_status = None

    def fetch_codebase_options(self):
        """Fetch available codebases from both Pinecone and GitHub."""
        if not st.session_state.repos_fetched:
            # Fetch from Pinecone
            pinecone_repos = self.embedding_handler.fetch_repos_from_pinecone()
            
            # Fetch from GitHub
            try:
                github_user = self.github.get_user("sheicky")  # Your GitHub username
                github_repos = [repo.html_url for repo in github_user.get_repos()]
                
                # Combine both lists
                st.session_state.repos = list(set(pinecone_repos + github_repos))
                st.session_state.repos_fetched = True
            except Exception as e:
                self.logger.error(f"Error fetching GitHub repos: {str(e)}")
                st.error("Failed to fetch GitHub repositories")

    def process_new_repository(self, github_url: str):
        """Process a new repository and store it in Pinecone."""
        progress_placeholder = st.empty()
        try:
            # 1. Clone and process repository
            progress_placeholder.info("🔄 Cloning repository...")
            files_content = self.repo_handler.process_repository(github_url)
            progress_placeholder.info("✅ Repository cloned successfully")

            # 2. Process files and create documents
            progress_placeholder.info("📊 Processing files...")
            documents = []
            for file_content in files_content:
                docs = self.embedding_handler.process_file_content(file_content)
                documents.extend(docs)
            progress_placeholder.info(f"✅ Processed {len(documents)} code chunks")

            # 3. Generate embeddings and upload to Pinecone
            progress_placeholder.info("🔍 Generating embeddings and uploading to Pinecone...")
            self.embedding_handler.upload_repo_to_pinecone(documents, github_url)
            progress_placeholder.success("✅ Repository added to Pinecone successfully!")

            # 4. Update session state
            if github_url not in st.session_state.repos:
                st.session_state.repos.append(github_url)
            st.session_state.selected_codebase = github_url
            
            # 5. Load initial context
            progress_placeholder.info("🔄 Loading initial context...")
            st.session_state.current_context = self.embedding_handler.load_repo_context(github_url)
            progress_placeholder.success("✅ Ready to answer questions about the codebase!")

            return True

        except Exception as e:
            progress_placeholder.error(f"❌ Error processing repository: {str(e)}")
            self.logger.error(f"Error processing repository: {str(e)}")
            return False

    def handle_codebase_selection(self):
        """Handle the codebase selection interface."""
        self.fetch_codebase_options()

        st.markdown("### 📚 Select an existing codebase")
        selected = st.selectbox(
            "Choose from existing codebases:",
            options=st.session_state.repos,
            index=st.session_state.repos.index(st.session_state.selected_codebase)
            if st.session_state.selected_codebase in st.session_state.repos
            else 0
        )

        st.markdown("### 🔄 Add a new repository")
        with st.form("new_repo_form"):
            github_url = st.text_input(
                "Enter GitHub repository URL:",
                placeholder="https://github.com/username/repo"
            )
            submit_button = st.form_submit_button("Process Repository", use_container_width=True)
            
            if submit_button and github_url:
                if self.process_new_repository(github_url):
                    st.rerun()

        if selected and selected != st.session_state.selected_codebase:
            progress_placeholder = st.empty()
            try:
                progress_placeholder.info("🔄 Loading context from Pinecone...")
                st.session_state.selected_codebase = selected
                st.session_state.current_context = self.embedding_handler.load_repo_context(selected)
                progress_placeholder.success("✅ Context loaded successfully!")
            except Exception as e:
                progress_placeholder.error(f"Error loading context: {str(e)}")

    def handle_model_selection(self):
        """Handle the AI model selection interface."""
        model_options = ["Google Gemini"]  # Added Llama to the options
        selected_model = st.selectbox(
            "Select AI model:",
            options=model_options,
            index=model_options.index(st.session_state.selected_model)
        )

        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model

    def display_chat_history(self):
        """Display the chat history."""
        for message in st.session_state.messages:
            with st.chat_message(message.role):
                st.markdown(message.content)
                st.caption(f"Sent at: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    def process_chat_input(self, prompt: str, image: Optional[str]):
        """Process chat input and generate response."""
        try:
            if not st.session_state.selected_codebase:
                st.error("Veuillez d'abord sélectionner un dépôt.")
                return
            
            response = self.embedding_handler.perform_rag(
                query=prompt,
                repo_url=st.session_state.selected_codebase,
                selected_model=st.session_state.selected_model
            )

            # Add messages to chat history
            st.session_state.messages.append(Message(
                role="user",
                content=prompt,
                timestamp=datetime.now()
            ))
            st.session_state.messages.append(Message(
                role="assistant",
                content=response,
                timestamp=datetime.now()
            ))

            # Display the new message
            with st.chat_message("assistant"):
                st.markdown(response)
                st.caption(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            self.logger.error(f"Error processing chat: {str(e)}")
            st.error("Failed to generate response. Please try again.")

    def display_chat_interface(self):
        """Display the chat interface and handle interactions."""
        for message in st.session_state.messages:
            avatar = "🤖" if message.role == "assistant" else "👤"
            st.markdown(f"""
                <div class="message-container {message.role}">
                    <div class="message-avatar">{avatar}</div>
                    <div class="message-content">
                        {message.content}
                        <div style="font-size: 0.75rem; color: var(--color-text-secondary); margin-top: 0.5rem;">
                            {message.timestamp.strftime('%H:%M')}
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Chat input
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        prompt = st.chat_input("Message CodeChat AI...")
        if prompt:
            self.process_chat_input(prompt, None)
        st.markdown('</div>', unsafe_allow_html=True)

    def run(self):
        """Run the Streamlit application."""
        st.set_page_config(
            page_title="CodeChat AI",
            page_icon="💻",
            layout="wide"
        )

        # Sidebar
        with st.sidebar:
            st.title("💻 CodeChat AI")
            st.caption("Your AI-powered code companion")
            
            # Repository Settings
            st.markdown("### Repository Settings")
            self.handle_codebase_selection()
            
            # Model Settings
            st.markdown("### Model Settings")
            self.handle_model_selection()
            
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
            
            st.divider()
            st.caption("❤️ Made with love by Sheick")

        # Main chat area
        if st.session_state.selected_codebase:
            st.markdown(f"""
            ### Active Codebase
            ```
            {st.session_state.selected_codebase}
            ```
            """)
            
            # Chat interface
            self.display_chat_interface()
        else:
            st.markdown("""
            ### 👋 Welcome to CodeChat AI!
            
            To get started:
            1. Select an existing codebase from the sidebar
            2. Or add a new repository by providing its GitHub URL
            3. Start asking questions about the code!
            
            Your AI assistant is ready to help you understand and work with the codebase.
            """)

def main():
    """Entry point of the application."""
    app = CodebaseRAGApp()
    app.run()

if __name__ == "__main__":
    main()