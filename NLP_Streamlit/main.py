# import InstructorEmbedding
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import SpacyEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil
# import re
import nltk
from nltk.corpus import stopwords
import spacy
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# import chromadb.utils.embedding_functions as embedding_functions
from unstructured.partition.pdf import partition_pdf
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# import string

# from chroma.embeddings import Embeddings


os.environ['OPENAI_API_KEY'] = '[API_Key]'

DATA_PATH = "data/finalBooks"
CHROMA_PATH = "chromaV3"

nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")


def partition_doc():
    raw_pdf_elements = partition_pdf(
        filename= DATA_PATH + "/docmerged.pdf",
        extract_images_in_pdf=False,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=6000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
    )

    # Categorize by type
    tables = []
    texts = []

    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))

    print(len(tables), " ", len(texts))
    # Prompt
    prompt_text = """You are an assistant tasked with summarizing tables and text. \
    Give a concise summary of the table or text. Table or text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

    return table_summaries, texts


def preprocess_text(text):
    # Tokenization and POS tagging using SpaCy
    doc = nlp(text)

    # Filtering out tokens based on POS tags and dependency parsing
    filtered_tokens = []
    for token in doc:
        if token.pos_ not in ["SPACE", "X"]:
            if token.dep_ not in ["det", "punct"]:
                filtered_tokens.append(token.text.lower())

    # Stopword removal
    filtered_tokens = [token for token in filtered_tokens if token not in stopwords.words('english')]

    # Lemmatization
    lemmatized_tokens = [token.lemma_ for token in nlp(" ".join(filtered_tokens))]

    return " ".join(lemmatized_tokens)


def load_documents():
    # loader = PyPDFDirectoryLoader("data/books/")
    loader = PyPDFLoader("data/books/symp.pdf")
    documents = loader.load()
    return documents


def split_text(table_summaries: list[str], textss: list[str]):
    # for doc in documents:
    #     doc.page_content = preprocess_text(doc.page_content)
    all_text = table_summaries + textss

    # Preprocess the text
    preprocessed_text = [preprocess_text(text) for text in all_text]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )

    chunks = []
    for text in preprocessed_text:
        # Split the text into chunks
        text_chunks = text_splitter.split_text(text)
        for chunk_text in text_chunks:
            # Create a new Document object for each chunk
            chunk_document = Document(page_content=chunk_text)
            chunks.append(chunk_document)

    print(f"Split {len(all_text)} texts into {len(chunks)} chunks.")
    # chunks = text_splitter.split_documents(documents)
    # print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    #
    # document = chunks[10]
    # print(document.page_content)
    # print(document.metadata)

    return chunks


def get_document_embeddings(documents, embedding_model):
    embeddings = []
    for doc in documents:
        embeddings.append(embedding_model(doc.page_content).vector)
    return embeddings


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


# documents = load_documents()
tableSummary, justText = partition_doc()
chunks = split_text(tableSummary, justText)
save_to_chroma(chunks)
