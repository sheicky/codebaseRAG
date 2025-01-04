import logging
from git import Repo
import os
from pathlib import Path
import shutil
from typing import List, Optional
from .models import FileContent

class RepositoryHandler:
    """
    Handles repository operations including cloning and file processing.
    
    This class manages GitHub repository operations for file reading operations.
    """

    SUPPORTED_EXTENSIONS = {
        ".py", ".js", ".tsx", ".jsx", ".ts", 
        ".java", ".cpp", ".md", ".json"
    }

    IGNORED_DIRS = {
        ".git", ".github", "dist", "__pycache__", 
        ".next", ".env", ".vscode", "node_modules", 
        ".venv", "h5"
    }

    def __init__(self, base_dir: str = "repos", logging_level: int = logging.INFO):
        """Initialize the repository handler."""
        self.logger = self._setup_logger(logging_level)
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.repo_path: Optional[Path] = None

    def _setup_logger(self, logging_level: int) -> logging.Logger:
        """Set up logging configuration."""
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

    def clone_repository(self, repo_url: str) -> Path:
        """Clone a GitHub repository."""
        try:
            repo_name = repo_url.split("/")[-1]
            self.repo_path = self.base_dir / repo_name
            
            # Nettoyer le dossier s'il existe
            if self.repo_path.exists():
                self._safe_remove(self.repo_path)
            
            self.logger.info(f"Cloning repository: {repo_url}")
            Repo.clone_from(repo_url, str(self.repo_path))
            return self.repo_path

        except Exception as e:
            self.logger.error(f"Failed to clone repository: {str(e)}")
            self._safe_remove(self.repo_path)
            raise

    def _safe_remove(self, path: Path) -> None:
        """Safely remove a directory and its contents."""
        try:
            if path and path.exists():
                # Rendre les fichiers accessibles en écriture avant suppression
                for item in path.rglob('*'):
                    if item.is_file():
                        item.chmod(0o666)
                    elif item.is_dir():
                        item.chmod(0o777)
                shutil.rmtree(path, ignore_errors=True)
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def get_file_content(self, file_path: Path) -> Optional[FileContent]:
        """
        Read content of a single file.
        
        Args:
            file_path: Path to the file

        Returns:
            FileContent object if successful, None otherwise
        """
        try:
            with open(file_path, mode='r', encoding='utf-8') as f:
                content = f.read()
                return FileContent(
                    name=str(file_path.relative_to(self.repo_path)),
                    content=content,
                    extension=file_path.suffix
                )
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            return None

    def get_main_files_content(self) -> List[FileContent]:
        """
        Get content of supported code files from the repository.
        
        Returns:
            List of FileContent objects
        """
        if not self.repo_path:
            raise ValueError("No repository path set")

        files_content = []

        for root, _, files in os.walk(self.repo_path):
            root_path = Path(root)
            
            if any(ignored_dir in root_path.parts for ignored_dir in self.IGNORED_DIRS):
                continue

            for file in files:
                file_path = root_path / file
                if file_path.suffix in self.SUPPORTED_EXTENSIONS:
                    content = self.get_file_content(file_path)
                    if content:
                        files_content.append(content)

        return files_content

    def process_repository(self, repo_url: str) -> List[FileContent]:
        """
        Process a repository: clone it and extract file contents.
        
        Args:
            repo_url: URL of the GitHub repository

        Returns:
            List of FileContent objects
        """
        try:
            self.clone_repository(repo_url)
            return self.get_main_files_content()
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up all temporary files."""
        try:
            self._safe_remove(self.repo_path)
            self._safe_remove(self.temp_dir)
            self.logger.info("Cleaned up repository directory")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
        finally:
            self.repo_path = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()