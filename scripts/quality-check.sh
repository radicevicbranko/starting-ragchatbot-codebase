#!/bin/bash

# Complete code quality check script for RAG chatbot project
set -e

echo "🚀 Starting complete code quality check..."

echo "📋 Step 1: Formatting code..."
./scripts/format.sh

echo "🔍 Step 2: Running linting and type checks..."
./scripts/lint.sh

echo "🧪 Step 3: Running tests..."
cd backend && uv run pytest tests/ -v

echo "✅ All quality checks passed!"