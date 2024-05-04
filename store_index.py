from src.helper import load_data, preprocess_text, chunk, embedding
import os
from dotenv import load_dotenv
load_dotenv()


# Get the path from environment variables
path = os.getenv('DATA_PATH')

# Load data from the specified path
documents = load_data(path)

# Preprocess the loaded data
for doc in documents:
    doc.page_content = preprocess_text(doc.page_content)

# Chunk the processed data
chunks = chunk(documents)

#embedding model
embedding=embedding()
from langchain_chroma import Chroma

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
db = Chroma.from_documents(chunks, embedding, persist_directory="./chroma_db-v1")