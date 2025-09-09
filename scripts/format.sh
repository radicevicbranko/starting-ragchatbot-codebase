#!/bin/bash

# Code formatting script for RAG chatbot project
set -e

echo "🔧 Formatting Python code with Black..."
uv run black backend/ main.py

echo "📋 Formatting and fixing imports with Ruff..."
uv run ruff format backend/ main.py
uv run ruff check --fix backend/ main.py

echo "✅ Code formatting completed!"