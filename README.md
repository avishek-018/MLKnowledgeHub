
# ML Knowledge Hub

ML Knowledge Hub is an MCP-enabled hybrid GraphRAG platform for discovering and reusing machine-learning assets such as research papers, models, datasets, and repositories.

## Motivation

Machine-learning teams often accumulate models, datasets, papers, and documentation across disconnected systems. This makes existing assets difficult to discover, compare, and reuse.

This project aims to combine:

- semantic retrieval
- Retrieval-Augmented Generation (RAG)
- knowledge graphs
- hybrid GraphRAG
- MCP-based tool access

to provide a unified ML knowledge-management assistant.

## Current Status

The project is currently in early development.

Implemented so far:

- Python project structure
- ML asset schema
- PDF text extraction
- document chunking
- end-to-end PDF ingestion pipeline
- unit tests with pytest

## Planned Architecture

```text
ML Assets
   |
   v
Document Ingestion
   |
   +--> Vector Embeddings --> Vector Database
   |
   +--> LLM Extraction --> Knowledge Graph
                            |
                            v
                     Hybrid Retrieval
                            |
                            v
                           LLM
                            |
                            v
                        MCP Server