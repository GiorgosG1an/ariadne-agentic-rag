# Ariadne (Αριάδνη) - Agentic RAG Academic Advisor

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/LlamaIndex-Workflows-purple)](https://www.llamaindex.ai/workflows)
[![LLM](https://img.shields.io/badge/Gemini-2.5_Flash-orange)](https://deepmind.google/technologies/gemini/)
[![Vector DB](https://img.shields.io/badge/Qdrant-Hybrid_Search-red)](https://qdrant.tech/)
[![Cache](https://img.shields.io/badge/Redis-Semantic_Cache-dc382d)](https://redis.io/)
[![Package Manager](https://img.shields.io/badge/uv-fast-yellow)](https://github.com/astral-sh/uv)

**Ariadne** is an advanced, Event-Driven Agentic Retrieval-Augmented Generation (RAG) system aiming to serve as the official AI Academic Advisor for the **Department of Informatics and Telecommunications (DIT) at the University of Peloponnese**. 

Just as the mythical Ariadne helped Theseus navigate the labyrinth, this AI system guides students through the complex "labyrinth" of academic regulations, course syllabi, schedules, and university announcements.

---

## Key Features & Architecture

Ariadne is built with a production-grade architecture focusing on low latency, high retrieval accuracy, and full observability.

* **Event-Driven Agentic Workflow:** Built on LlamaIndex Workflows, replacing rigid linear chains with a flexible, async event-driven architecture (Routing, Query Rewriting, Fallback mechanisms).
* **Semantic Caching:** Integrates `RedisVL` to cache LLM responses based on semantic similarity. Repeated or similar questions bypass the LLM and DB entirely, returning answers instantly (Sub-100ms response times).
* **Hybrid Search:** Utilizes **Qdrant** for hybrid retrieval, combining Dense embeddings (Gemini-2-Embedding) with Sparse vectors (BM25) for highly accurate context retrieval.
* **Smart Auto-Retrieval:** Features dynamic metadata filtering (e.g., filtering by semester or course category) via LlamaIndex's `VectorIndexAutoRetriever` driven by an intelligent routing LLM (Gemini 2.5 Flash-Lite).
* **Full Observability:** Instrumented with OpenTelemetry, streaming real-time traces and spans to **Arize Phoenix** for debugging, latency monitoring, and prompt tracking.
* **Resilient Data Pipelines:** Custom PDF parsers and web scrapers feed an automated, safe-mode data ingestion pipeline handling rate limits and API timeouts gracefully.

---

## Tech Stack

| Component | Technology |
| :--- | :--- |
| **Core Framework** | [LlamaIndex](https://www.llamaindex.ai/) (Workflows & Memory) |
| **LLMs & Embeddings** | Google Gemini 2.5 Flash, Flash-Lite, Gemini-2-Embeddings |
| **Vector Database** | [Qdrant](https://qdrant.tech/) |
| **Semantic Cache** | [Redis Stack](https://redis.io/docs/about/about-stack/) + `redisvl` |
| **Observability** | [Arize Phoenix](https://phoenix.arize.com/) (OpenTelemetry) |
| **User Interface** | [Chainlit](https://docs.chainlit.io/) |
| **Dependency Manager** | [uv](https://github.com/astral-sh/uv) by Astral |

---

## Project Structure

The repository follows a modern, professional `src/` layout:

```text
ariadne-agentic-rag/
├── data/                    # Raw & processed data (PDFs, JSONL, Markdown)
├── eval/                    # LLM-as-a-judge evaluation scripts & datasets
├── infrastructure/          # Docker compose configurations
├── pipelines/               # Offline ETL jobs
│   ├── scrapers/            # Web and announcement scrapers
│   ├── extractors/          # Semantic PDF parsers & schedule extractors
│   ├── processors/          # LLM-based metadata enrichment
│   └── ingestion.py         # Main script for populating Qdrant & Redis
├── src/ariadne/             # Core application package
│   ├── core/                # Configs, Tracing, Logging, and Dependencies
│   ├── infrastructure/      # Qdrant and Redis connection singletons
│   ├── agent/               # Event-Driven RAG Workflow & Prompts
│   └── ui/                  # Chainlit frontend application
├── compose.yaml             # Multi-container infrastructure stack
├── pyproject.toml           # uv project configuration and dependencies
└── README.md
```

## Getting Started
1. Prerequisites
- Python 3.12+
- Docker & Docker Compose
- [uv](https://github.com/astral-sh/uv) (The ultra-fast Python package installer)

2. Clone & Install
```bash
git clone https://github.com/GiorgosG1an/ariadne-agentic-rag.git
cd ariadne-agentic-rag

# Install dependencies and setup the virtual environment in milliseconds
uv sync
```

3. Setup Infrastructure
```bash
docker compose up -d
```
*Wait a few seconds for the health checks to pass (docker compose ps).*

4. Configuration
Copy the environment template and add your Google Gemini API key:
```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```
5. Data Ingestion (Populating the Vector DB)
To feed Ariadne with the university's data:
1. Ensure your data is placed inside the data/ directory.
2. Run the ingestion pipeline:

```bash
uv run pipelines/ingestion.py
```

6. Launch the Application
```bash
uv run chainlit run src/ariadne/ui/app.py
```

- UI: `http://localhost:8000`
- Tracing (Arize Phoenix): `http://localhost:6006`
- Cache Dashboard (RedisInsight): `http://localhost:8001`

## Contributing
Contributions, issues, and feature requests are welcome!

## License & Acknowledgments
Developed by [Giorgos Giannopoulos](https://github.com/GiorgosG1an) as undergraduate thesis project for the Department of Informatics and Telecommunications, University of Peloponnese.

This project is licensed under the **MIT License** - see the[LICENSE](LICENSE) file for details.