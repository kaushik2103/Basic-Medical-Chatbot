from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import streamlit as st
import ingest as ig

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
if you know the answer, just provide the medication, treatments and diagnosis.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def set_custom_prompt():
    # Prompt template for QA retrieval for each vectorstore

    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 3}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain


# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=500,
        temperature=0.5
    )
    return llm


# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa


# Output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({"query": query})
    return response.get("result")


# Streamlit code
def main():
    st.set_page_config(page_title="Medical ChatBot", page_icon=":robot:")
    st.title("Medical Treatment and Diagnosis ChatBot")
    st.write("Starting the bot...")
    st.write("Hi, Welcome to Medical ChatBot. What is your query?")
    query = st.text_input("Enter your query:")
    if query:
        answer = final_result(query)
        st.write("Bot's response:")
        st.write(answer)
    else:
        st.write("Please enter a query.")


if __name__ == "__main__":
    main()
