from openai import OpenAI
import streamlit as st
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import SpacyEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser

st.title("Apna Doctor")

os.environ['OPENAI_API_KEY'] = '[API_Key]'

CHROMA_PATH = "chroma_db-v(Harrison-mehta)"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
context_text = ""

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Perform similarity search
    results = db.similarity_search_with_relevance_scores(prompt, k=3)
    if len(results) == 0 or results[0][1] < 0.5:
        st.write("Unable to find matching results.")
    else:
        new_context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        context_text = "\n\n---\n\n".join([context_text, new_context_text])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        promptt = prompt_template.format(context=context_text, question=prompt)

        # Generate response using OpenAI
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                messages = [
                    {"role": m["role"], "content": promptt}
                    for m in st.session_state.messages
                ]
                model = ChatOpenAI() | StrOutputParser()
                response_text = model.invoke(promptt)
                text = f"{response_text}"
                st.write(text)

                # Load the model
                sources = [doc.metadata.get("source", None) for doc, _score in results]
                formatted_response = f"{response_text}"
                # st.write(response_text)
                st.session_state.messages.append({"role": "assistant", "content": formatted_response})
        # st.session_state.messages.append({"role": "assistant", "content": response})

        # Also I'm feeling headache sometimes & feeling weakness since many days