import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

import os

st.set_page_config(page_title="📄 DocumentGPT", page_icon="📄")
st.title("📄 DocumentGPT - RAG Chatbot")

# 사이드바
with st.sidebar:
    st.markdown("## 🔑 OpenAI API Key")
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
    st.markdown("## 📁 Upload File")
    uploaded_file = st.file_uploader("Upload a .pdf, .txt, or .docx file", type=["pdf", "txt", "docx"])
    st.markdown("---")
    st.markdown("🔗 [GitHub Repository](https://github.com/yourusername/your-repo)")

# API 키 체크
if not openai_api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 파일 업로드 후 임베딩
@st.cache_data(show_spinner="🔍 Embedding file...")
def embed_file(file):
    file_path = f"./.cache/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file.read())

    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=600, chunk_overlap=100)
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever()

# 프롬프트 설정
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using ONLY the following context. If you don't know the answer, just say you don't know.\n\nContext: {context}"),
    ("human", "{question}")
])

# 파일 업로드 되었을 때 실행
if uploaded_file:
    retriever = embed_file(uploaded_file)

    # 채팅 히스토리 출력
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["message"])

    # 사용자 입력
    user_input = st.chat_input("Ask something about your document...")

    if user_input:
        st.session_state["messages"].append({"role": "user", "message": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        llm = ChatOpenAI(temperature=0.1, openai_api_key=openai_api_key)
        chain = {
            "context": retriever | RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough()
        } | prompt | llm

        with st.chat_message("assistant"):
            response = chain.invoke(user_input)
            st.markdown(response)
            st.session_state["messages"].append({"role": "assistant", "message": response})
else:
    st.info("📄 파일을 업로드하면 챗봇이 작동합니다.")