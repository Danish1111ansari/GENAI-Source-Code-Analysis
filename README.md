# 🚀 GENAI Source Code Analysis

A powerful Retrieval-Augmented Generation (RAG) system that analyzes source code repositories using local AI models. This tool allows you to query any codebase using natural language without requiring expensive API keys.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-yellow)
![LangChain](https://img.shields.io/badge/LangChain-Framework-orange)
![Open Source](https://img.shields.io/badge/Open%20Source-Yes-brightgreen)

## ✨ Features

- **🔍 Code Repository Analysis**: Automatically clones and processes GitHub repositories
- **🤖 Local AI Models**: Uses free Hugging Face models (no API keys required)
- **💬 Natural Language Queries**: Ask questions about code in plain English
- **📚 Multi-language Support**: Java, Python, JavaScript, TypeScript, C++, and more
- **🔗 Smart Retrieval**: Semantic search with ChromaDB vector database
- **💾 Conversation Memory**: Maintains context across multiple questions
- **🌐 Web Interface**: User-friendly Flask web application
- **🆓 Completely Free**: No usage limits or subscription costs

## 🛠️ Tech Stack

### Backend
- **Python** - Primary programming language
- **Flask** - Web framework
- **LangChain** - LLM orchestration framework
- **ChromaDB** - Vector database for embeddings

### AI Models
- **sentence-transformers/all-MiniLM-L6-v2** - Text embeddings
- **microsoft/DialoGPT-medium** - Language model for conversations
- **Hugging Face Transformers** - Model inference

### Code Processing
- **GitPython** - Repository cloning and management
- **LangChain Document Loaders** - Multi-language code parsing
- **Recursive Text Splitting** - Intelligent code chunking



