#!/bin/bash

# Linting script for RAG chatbot project
set -e

echo "🔍 Running Ruff linting checks..."
uv run ruff check backend/ main.py

echo "🏷️  Running MyPy type checking..."
uv run mypy backend/ main.py

echo "✅ All linting checks completed!"