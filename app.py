from flask import Flask, render_template, jsonify, request
from src.prompt import PROMPT_TEMPLATE
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_transformers import LongContextReorder


import os
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

db = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db-v1")
retriever = db.as_retriever()


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form["msg"]
        input_query  = msg
        print(input_query)
        results = db.similarity_search_with_relevance_scores(input_query, k=7)
        if len(results) == 0 or results[0][1] < 0.7:
            print(f"Unable to find matching results.")
        # Reorder documents
        reordering = LongContextReorder()
        reorder_docs = reordering.transform_documents(results)
        context_text = "\n\n---\n\n".join([doc[0].page_content for doc in reorder_docs])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=input_query)
        model = ChatOpenAI() | StrOutputParser()
        response_text =model.invoke(prompt)
        # print("Response : ", result["result"])
        # sources = [doc.metadata.get("source", None) for doc, _score in reorder_docs]
        # formatted_response = f"Response: {response_text }\nSources: {sources}"
        print(response_text)
        return response_text
    
    except Exception as e:
        return "An error occurred during chat processing. Please try again later.", 500
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)


