import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template

#C:/Users/simon.rudat/OneDrive - WHU/Promotion/1st Project/Multi PDF Python Script
#https://www.youtube.com/watch?v=dXxQ0LR-3Hg
#Limitations:
#1) Abstraction level -> Model doesn't know that it was provided with a sample of papers
#2) Ability to do data synthesis -> Models tends to say it cannot do the task. Maybe, only useful to information queries

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # TODO: Add your OpenAI API key in quotation marks
    embeddings = OpenAIEmbeddings(openai_api_key="")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key="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")
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
    

st.set_page_config(page_title="Chat wth multiple PDFs", page_icon=":books:")

st.write(css, unsafe_allow_html=True)

# Initialize session variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

st.header("Chat with multiple PDFs :books:")
user_question = st.text_input("Ask a question about your documents:")
if user_question:
    handle_userinput(user_question)

#st.write(user_template.replace("{{MSG}}", "user"), unsafe_allow_html=True)
#st.write(bot_template.replace("{{MSG}}", "bot"), unsafe_allow_html=True)
    
with st.sidebar:
    st.subheader("Your documents")
    pdf_docs=st.file_uploader("Uplolad your PDFs here and click on 'Process'", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing"):
            # get pdf test
            raw_text=get_pdf_text(pdf_docs)
            
            
            # get the text chunks
            text_chunks = get_text_chunks(raw_text)
            #st.write(text_chunks)
            
            # create vector store
            vectorstore = get_vectorstore(text_chunks)
            
            # create conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore)
            