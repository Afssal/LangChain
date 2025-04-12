from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
import streamlit as st
from pathlib import Path
import os

#load llm
llm = OllamaLLM(model='llama3.2:1b')

#load embedding
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

#create tmp directory
if not os.path.isdir('tmp'):
    os.mkdir('tmp')

#upload file and save it on tmp folder
file = st.file_uploader("Upload pdf")
if file:
    save_folder = './tmp'
    save_path = Path(save_folder,file.name)
    with open(save_path,mode='wb') as w:
        w.write(file.getvalue())

    #load pdf to pypdfloader
    loader = PyPDFLoader(f'./tmp/{file.name}')
    pages = loader.load()

    #split document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(pages)

    #create vector database and splitted data on database
    persist_directory = 'docs/chroma/'

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    #user query
    question = st.text_input("Enter the question")
    # docs = vectordb.similarity_search(question,k=3)

    # Build prompt
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Run chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    result = qa_chain({"query": question})

    st.write(result['result'])
    # print(result["result"])

