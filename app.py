import streamlit as st
import pickle
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
import os

# Sidebar contents
with st.sidebar:
    st.title("LLM Chat App")
    st.markdown('''
    ## About
    **This app is an LLM-powered chatbot built using:**
    - [Streamlit](https://streamlit.io/)
    - [Langchain](https://python.langchain.com/)
    ''')
    add_vertical_space(5)
    st.write('Made by SB')

load_dotenv()

def main():
    st.header(":rainbow[Chat with PDF 💬]")

    # upload pdf files
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        # embeddings
        # Using a real embedding model for better performance
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Accept user query
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            # Make sure to set your GROQ_API_KEY in your .env file
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                st.error("GROQ_API_KEY is not set. Please add it to your .env file.")
                st.stop()
                
            llm = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.invoke({"input_documents": docs, "question": query})

            st.write("### Answer:")
            st.write(response["output_text"])

if __name__ == '__main__':
    main()