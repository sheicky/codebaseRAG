from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime

@dataclass
class FileContent:
    """
    Data class representing a file's content and metadata.
    
    Attributes:
        name: The name/path of the file relative to the repository root
        content: The actual content of the file
        extension: The file extension
        last_modified: Timestamp of last modification
    """
    name: str
    content: str
    extension: str
    last_modified: datetime = datetime.now()

@dataclass
class CodeChunk:
    """
    Data class representing a chunk of code with its metadata.
    
    Attributes:
        content: The actual code content
        metadata: Additional metadata about the chunk
    """
    content: str
    metadata: Dict[str, Any]

@dataclass
class Message:
    """
    Data class representing a chat message.
    
    Attributes:
        role: The role of the message sender (user/assistant)
        content: The content of the message
        timestamp: When the message was created
    """
    role: str
    content: str
    timestamp: datetime = datetime.now()