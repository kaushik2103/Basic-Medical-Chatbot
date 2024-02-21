import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA


class MedicalChatBot:
    def __init__(self):
        self.DB_FAISS_PATH = 'vectorstore/db_faiss'
        self.custom_prompt_template = """Use the following pieces of information to answer the user's question.
        Don't give unnecessary statements.
        If you don't know the answer, just say that you don't know.
        if you know the answer, just provide the medication, treatments and diagnosis.

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else.
        Helpful answer:
        """
        self.qa = self.qa_bot()

    def set_custom_prompt(self):
        prompt = PromptTemplate(template=self.custom_prompt_template, input_variables=['context', 'question'])
        return prompt

    def retrieval_qa_chain(self, llm, prompt, db):
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff',
                                               retriever=db.as_retriever(search_kwargs={'k': 3}),
                                               return_source_documents=True,
                                               chain_type_kwargs={'prompt': prompt})
        return qa_chain

    def load_llm(self):
        llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML", model_type="llama", max_new_tokens=500,
                            temperature=0.5)
        return llm

    def qa_bot(self):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={'device': 'cpu'})
        db = FAISS.load_local(self.DB_FAISS_PATH, embeddings)
        llm = self.load_llm()
        qa_prompt = self.set_custom_prompt()
        qa = self.retrieval_qa_chain(llm, qa_prompt, db)
        return qa

    def final_result(self, query):
        response = self.qa({"query": query})
        return response.get("result")

    def run(self):
        st.set_page_config(page_title="Medical ChatBot", page_icon=":robot:")
        st.title("Medical Treatment and Diagnosis ChatBot")
        st.write("Starting the bot...")
        st.write("Hi, Welcome to Medical ChatBot. What is your query?")
        query = st.text_input("Enter your query:")
        if query:
            answer = self.final_result(query)
            st.write("Bot's response:")
            st.write(answer)
        else:
            st.write("Please enter a query.")


if __name__ == "__main__":
    bot = MedicalChatBot()
    bot.run()
