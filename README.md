<div align="center">

# 🤖 CodeChat AI

![image](https://github.com/user-attachments/assets/0ece57e2-39df-498f-a5b9-19deee0dca56)


[![GitHub license](https://img.shields.io/github/license/sheick/codechat-ai)](https://github.com/sheick/codechat-ai/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28.2-FF4B4B)](https://streamlit.io)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

_Your AI-Powered Code Companion for Intelligent Repository Analysis_

[Demo](https://codechat-ai.demo.com) • [Documentation](https://docs.codechat-ai.com) • [Report Bug](https://github.com/sheick/codechat-ai/issues) • [Request Feature](https://github.com/sheick/codechat-ai/issues)

</div>

---

## 🎯 Overview

CodeChat AI revolutionizes code understanding by combining the power of RAG (Retrieval-Augmented Generation) with advanced language models. It enables developers to have meaningful conversations about their codebase while maintaining deep context across multiple repositories.

<div align="center">
<img src="https://your-demo-gif-url.gif" alt="Youtube Demo" width="600"/>
</div>

## ✨ Key Features

🔄 **Smart Repository Integration**

- Seamless GitHub repository connection
- Automatic code analysis and indexing
- Support for multiple programming languages

🧠 **Advanced Context Understanding**

- RAG-powered code comprehension
- Maintains context across conversations
- Deep understanding of code structure

🤖 **Multi-Model AI Support**

- Google Gemini integration
- Extensible model architecture
- Optimized for code understanding

💬 **Interactive Development Experience**

- Natural language code queries
- Contextual code suggestions
- Real-time response generation

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have:

- Python 3.8 or higher

### ⚡ Quick Install

1. **Clone & Setup**

```bash
# Clone the repository
git clone https://github.com/sheicky/codechat-ai.git
cd codechat-ai

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Unix/macOS
# or
.\venv\Scripts\activate  # Windows
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Configure Environment**

```bash
# Create .env file
cp .env.example .env

# Add your API keys to .env
PINECONE_API_KEY=your_pinecone_key
GOOGLE_API_KEY=your_google_key
GITHUB_TOKEN=your_github_token
```

4. **Launch Application**

```bash
streamlit run rag_app.py
```

## 🏗️ Architecture

```mermaid
graph TD
    A[GitHub Repository] --> B[Repository Handler]
    B --> C[Code Processor]
    C --> D[Embedding Generator]
    D --> E[Pinecone Vector Store]
    E --> F[RAG Engine]
    F --> G[LLM Interface]
    G --> H[User Interface]
```

## 🛠️ Technology Stack

| Category            | Technologies                                                                                                                                                               |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Frontend**        | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)                                                                      |
| **Backend**         | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)                                                                               |
| **AI/ML**           | ![LangChain](https://img.shields.io/badge/LangChain-121212?style=flat) ![Google Gemini](https://img.shields.io/badge/Gemini-4285F4?style=flat&logo=google&logoColor=white) |
| **Vector Store**    | ![Pinecone](https://img.shields.io/badge/Pinecone-121212?style=flat)                                                                                                       |
| **Version Control** | ![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white)                                                                                        |

## 📈 Performance

- **Frontend**: Streamlit
- **Embeddings**: Sentence Transformers
- **Vector Store**: Pinecone
- **LLM**: Google Gemini
- **Code Processing**: LangChain
- **Version Control**: Git


```bash
# Development workflow
git checkout -b feature/amazing-feature
git commit -m 'feat: add amazing feature'
git push origin feature/amazing-feature
```



## 👏 Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain) for RAG implementation
- [Pinecone](https://www.pinecone.io/) for vector storage
- [Google Gemini](https://deepmind.google/technologies/gemini/) for AI capabilities
- [Streamlit](https://streamlit.io/) for the UI framework

## 📬 Contact & Support

- LinkedIn: [Sheick](https://www.linkedin.com/in/pensas/)

<div align="center">

Made with ❤️ by Sheick | Copyright © 2024

</div>
