# langchain
# langchain-openai
# ctransformers
# sentence-transformers
# langchain-chroma
# pandas
# nltk
# spacy
# PyPDF2
# flask
# python-dotenv
# -e .

import os

# Install required packages
os.system("pip install -U langchain-community")
os.system("pip install langchain langchain-openai")
os.system("pip install ctransformers sentence-transformers langchain-chroma")
os.system("pip install pandas nltk spacy PyPDF flask")
os.system("pip install --upgrade --quiet sentence-transformers langchain-chroma langchain langchain-openai > NUL")
# os.system("pip install --upgrade --quiet sentence-transformers langchain-chroma langchain langchain-openai")