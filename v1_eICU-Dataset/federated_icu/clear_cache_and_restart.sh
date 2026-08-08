#!/usr/bin/env bash
# Run this from the fedicu_updated/ directory before restarting the app.
# It clears all Python bytecode caches so your updated .py files are used.

echo "Clearing Python __pycache__ directories..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "Done. Now restart the Flask app: python run.py"
