import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import SpacyEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.schema import Document
import os
import shutil
import re
import string
import nltk
import spacy
from nltk.corpus import stopwords
from google.colab import drive
drive.mount("/content/drive")
path="/content/drive/MyDrive/Colab Notebooks/data"


nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

os.environ['OPENAI_API_KEY'] = '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'

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def preprocess_text(text):
    # Tokenization
    tokens = re.findall(r'\b\w+\b', text)
    # Lowercasing
    tokens = [token.lower() for token in tokens]
    # Stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatization
    tokens = [token.lemma_ for token in nlp(" ".join(tokens))]
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    return " ".join(tokens)


def main():
    # Load documents from PDF
    loader = DirectoryLoader( path, glob="Medical_book.pdf", loader_cls=PyPDFLoader )
    documents = loader.load()

    # Preprocess each document
    for doc in documents:
        doc.page_content = preprocess_text(doc.page_content)

    # Split the preprocessed documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Save preprocessed chunks to Chroma
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

    # Create CLI.
    # parser = argparse.ArgumentParser()
    # parser.add_argument("query_text", type=str, help="The query text.")
    # args = parser.parse_args()
    query_text = "I'm having fever since two weeks, also sometimes vomiting with watery eyes, what could be the disease?"

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.invoke(prompt)

    # model_name = "gpt2"
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # model = GPT2LMHeadModel.from_pretrained(model_name)
    #
    # input_ids = tokenizer.encode(prompt, return_tensors="pt")
    #
    # # Generate response
    # output = model.generate(input_ids, max_length=1000, num_return_sequences=1, temperature=0.9)
    #
    # response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(response_text)

    # output = replicate.run(
    #     "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    #     input={
    #         "prompt": prompt,
    #         "temperature": 0.1,
    #         "max_length": 2500,
    #         "top_p": 0.9
    #     }
    # )
    #
    # # The meta/llama-2-70b-chat model can stream output as it's running.
    # # The predict method returns an iterator, and you can iterate over that output.
    # for item in output:
    #     # https://replicate.com/meta/llama-2-70b-chat/api#output-schema
    #     print(item, end="")

    # Generate LLM response
    # output = replicate.run(
    #     'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5',  # LLM model
    #     input={"prompt": f"{prompt} Assistant: ",  # Prompts
    #            "temperature": 0.1, "top_p": 0.9, "max_length": 1000, "repetition_penalty": 1})  # Model para

    # full_response = ""
    #
    # for item in output:
    #     full_response += item
    #
    # print(full_response)


    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
    # model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")
    #
    # #input_text = "Write me a poem about Machine Learning."
    # input_ids = tokenizer(prompt, return_tensors="pt")
    #
    # outputs = model.generate(**input_ids, max_length=750)
    # print(tokenizer.decode(outputs[0]))


    # Load the model
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if _name_ == "_main_":
    main()
