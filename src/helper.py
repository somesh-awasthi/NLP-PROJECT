
# Load documents from PDF
# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
def load_data(path):
    loader = DirectoryLoader(path, glob="disease-handbook-complete.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


# pre-processing
import re
import nltk
import spacy
import string
from nltk.corpus import stopwords

nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Tokenization and POS tagging using SpaCy
    doc = nlp(text)

    # Filtering out tokens based on POS tags and dependency parsing
    filtered_tokens = [token.text.lower() for token in doc if token.pos_ not in ["SPACE", "X"] and token.dep_ not in ["det", "punct"]]

    # Stopword removal
    filtered_tokens = [token for token in filtered_tokens if token not in stopwords.words('english')]

    # Lemmatization
    lemmatized_tokens = [token.lemma_ for token in nlp(" ".join(filtered_tokens))]

    return " ".join(lemmatized_tokens)


#chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Split the preprocessed documents
def chunk(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,  
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)   
    return chunks

#embedding
from langchain_openai import OpenAIEmbeddings
def embedding():
    return OpenAIEmbeddings()