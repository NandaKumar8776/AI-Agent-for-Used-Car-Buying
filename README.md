# AI Agent- for Used Car Buying

## Overview

This project is an advanced, modular Retrieval-Augmented Generation (RAG) system designed to answer questions about used car buying. It leverages state-of-the-art LLMs, hybrid search (BM25 + vector), and web crawling to provide accurate, context-aware responses. The system is orchestrated using [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain).

## Features

- **Supervisor Node:** Classifies user queries and routes them to the appropriate processing node.
- **LLM Node:** Handles general questions using a language model.
- **RAG Node:** Answers questions using a hybrid search (BM25 + vector) over a provided PDF guide.
- **Web Crawler Node:** Scrapes a car dealership inventory website and answers queries using hybrid retrieval over the scraped data.
- **Prompt Engineering:** Custom prompts for each node, stored in the `prompts/` directory.


## Directory Structure

```
AI-Agent-for-Used-Car-Buying/
  ├── config/           # Environment setup
  ├── data/             # Source data (PDF guide)
  ├── graph/            # Workflow and node logic
  ├── memory/           # Vector store and BM25 logic
  ├── prompts/          # Prompt templates for each node
  ├── testing/          # Test scripts (placeholders)
  ├── tools/            # Pipelines for LLM, RAG, web crawling
  ├── utils/            # Helper functions
  ├── main.py           # Entry point example
  ├── requirements.txt  # Dependencies
  └── README.md         # Project documentation
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd AI-Agent-for-Used-Car-Buying
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Copy your API keys and prompt file paths into a `.env` file in the root directory. Required variables (see `config/env_setup.py`):


4. **Data:**
   - Ensure `data/The-Ultimate-Used-Car-Buying-Guide.pdf` is present (already included).

## Usage

- **Run the example workflow:**
  ```bash
  python main.py
  ```
  This will process a sample question about used car buying and print the system's response.

- **Custom Queries:**
  Modify the `question` variable in `main.py` to test different queries.


## Testing

- Placeholder test scripts are in the `testing/` directory. Add your own tests as needed.

## Dependencies

See `requirements.txt` for the full list


