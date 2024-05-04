import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Execute initial setup commands
os.system("pip install langchain langchain-openai")
os.system("pip install ctransformers sentence-transformers langchain-chroma")
os.system("pip install pandas nltk spacy PyPDF")
os.system("pip install --upgrade --quiet sentence-transformers langchain-chroma langchain langchain-openai > /dev/null")

# os.system("python -m spacy download en_core_web_sm")
# os.system("%pip install --upgrade --quiet sentence-transformers langchain-chroma langchain langchain-openai")


list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "research/NLP_Project_V1_2.ipynb",
    "app.py",
    "store_index.py",
    "static/.gitkeep",
    "templates/chat.html"

]
# along with this also do the `pip install flask`

for filepath in list_of_files:
   filepath = Path(filepath)
   filedir, filename = os.path.split(filepath)

   if filedir !="":
      os.makedirs(filedir, exist_ok=True)
      logging.info(f"Creating directory; {filedir} for the file {filename}")

   if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
      with open(filepath, 'w') as f:
         pass
         logging.info(f"Creating new file: {filepath}")

   else:
      logging.info(f"{filename} is already there")
      
      
    