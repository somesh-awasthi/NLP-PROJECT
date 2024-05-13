from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import SpacyEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
# import openai

os.environ['OPENAI_API_KEY'] = '[API_Key]'

CHROMA_PATH = "chromaV1"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    context_text = ""
    while True:
        query_text = input("Enter your query (type 'quit' to exit): ")

        if query_text.lower() == 'quit':
            break

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        if len(results) == 0 or results[0][1] < 0.5:
            print(f"Unable to find matching results.")
            return

        new_context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        context_text += "\n\n---\n\n" + new_context_text
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        print(prompt)

        model = ChatOpenAI()
        response_text = model.invoke(prompt)

        # Load the model
        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        print(formatted_response)


    # query_text = "I'm having fever since two weeks, also sometimes vomiting with watery eyes, what could be the disease?"


if __name__ == "__main__":
    main()