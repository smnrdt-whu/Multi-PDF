import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

#https://www.youtube.com/watch?v=dXxQ0LR-3Hg
#Limitations:
#1) Abstraction level -> Model doesn't know that it was provided with a specific knowledge base
#2) Ability to do data synthesis -> Models tends to say it cannot do tasks related to data synthesis

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    # Chunk size and overlap is a hyperparameter that can be tuned to improve performance
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
    token_split_texts = []
    for text in chunks:
        token_split_texts += token_splitter.split_text(text)
    return token_split_texts

def get_vectorstore(text_chunks):
    if model=='Hugging Face (free)':
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=key)
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    if model=='Hugging Face (free)':
        llm = HuggingFaceHub(huggingfacehub_api_token=key, repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    else:
        llm = ChatOpenAI(openai_api_key=key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    
def process_documents():
    # get pdf test
    raw_text=get_pdf_text(pdf_docs)
    # get the text chunks
    text_chunks = get_text_chunks(raw_text)
    # create vector store
    vectorstore = get_vectorstore(text_chunks)
    # create conversation chain
    st.session_state.conversation = get_conversation_chain(vectorstore)
    
def disable():
    st.session_state.disabled = True

st.set_page_config(page_title="Chat wth multiple PDFs", page_icon=":books:")

st.write(css, unsafe_allow_html=True)

# Initialize session variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
if "disabled" not in st.session_state:
    st.session_state.disabled = False

st.header("Chat with multiple PDFs :books:")

# Pick preferred model
model = st.radio(label="Pick your preferred model:", options=['Hugging Face (free)', 'OpenAI (high performance)'], disabled=st.session_state.disabled)
# API key input
key = st.text_input("Enter your API key:", on_change=disable, disabled=st.session_state.disabled)
if key:
    st.write("✅")

#st.subheader("Your documents")
pdf_docs=st.file_uploader("Uplolad your PDFs here:", accept_multiple_files=True)
if pdf_docs:
    st.write("✅")

# Prompt input
user_question = st.text_input("Ask a question about your documents:", on_change=process_documents)
if user_question:
    handle_userinput(user_question)