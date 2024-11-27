import os
import json
import logging
import streamlit as st
import tiktoken
from loguru import logger

# 최신 LangChain 경로로 업데이트
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="dirchat.log",
)

def main():
    st.set_page_config(
        page_title="DirChat",
        page_icon=":books:",
    )

    st.title("_Private Data :red[QA Chat]_ :books:")

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "안녕하세요! 업로드된 문서에 대해 질문해주세요."}
        ]
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False

    with st.sidebar:
        # 파일 업로드
        uploaded_files = st.file_uploader(
            "파일 업로드",
            type=["pdf", "docx", "pptx", "json"],
            accept_multiple_files=True,
        )

        # 업로드 파일 검증
        if uploaded_files:
            if len(uploaded_files) > 5:
                st.warning("최대 5개의 파일만 업로드할 수 있습니다.")
                uploaded_files = uploaded_files[:5]
            uploaded_files = [
                f for f in uploaded_files if f.size <= 10 * 1024 * 1024
            ]  # 10MB 제한

        openai_api_key = st.text_input("OpenAI API Key", type="password")
        process = st.button("문서 처리")

    if process:
        if not openai_api_key:
            st.info("OpenAI API 키를 입력해주세요.")
            return

        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vector_store = get_vectorstore(text_chunks)
        st.session_state.conversation = get_conversation_chain(vector_store, openai_api_key)
        st.session_state.processComplete = True

    # 이전 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            if not st.session_state.conversation:
                st.markdown("문서를 처리한 후 질문해주세요.")
                return

            with st.spinner("답변 생성 중..."):
                result = st.session_state.conversation({"question": query})
                response = result["answer"]
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

# 기타 함수 정의 (변경 없음)
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    # 파일 로더 정의 및 텍스트 추출
    ...

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=100, length_function=tiktoken_len
    )
    return text_splitter.split_documents(text)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
    )
    return FAISS.from_documents(text_chunks, embeddings)

def get_conversation_chain(vector_store, openai_api_key):
    llm = ChatOpenAI(
        openai_api_key=openai_api_key, 
        model_name="gpt-3.5-turbo", 
        temperature=0
    )
    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vector_store.as_retriever(),
        memory=ConversationBufferMemory(),
    )

if __name__ == "__main__":
    main()
