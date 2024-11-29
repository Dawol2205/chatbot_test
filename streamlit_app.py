import streamlit as st
import tiktoken
import json
import os
import requests
from loguru import logger
import pickle
from datetime import datetime

# Updated LangChain imports
from langchain_community.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback

# GitHub 저장소 정보
GITHUB_REPO = "K-MarkLee/AI_8_CH-3_LLM-RAG_AI_Utilizatioon_App"
GITHUB_BRANCH = "Mark"
GITHUB_PATH = "personal_work/이승열/food_db"
FAISS_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/{GITHUB_PATH}/index.faiss"
PKL_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/{GITHUB_PATH}/index.pkl"

def validate_api_key(api_key):
    """OpenAI API 키 형식 검증"""
    return api_key and len(api_key) > 20

def download_vector_store():
    """GitHub에서 FAISS 벡터 저장소 다운로드"""
    try:
        # FAISS 파일 다운로드
        faiss_response = requests.get(FAISS_URL)
        if faiss_response.status_code != 200:
            return False, f"FAISS 파일 다운로드 실패: {faiss_response.status_code}"

        # PKL 파일 다운로드
        pkl_response = requests.get(PKL_URL)
        if pkl_response.status_code != 200:
            return False, f"PKL 파일 다운로드 실패: {pkl_response.status_code}"

        # 임시 파일로 저장
        with open("index.faiss", "wb") as f:
            f.write(faiss_response.content)
        with open("index.pkl", "wb") as f:
            f.write(pkl_response.content)

        # 임베딩 초기화
        embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # FAISS 벡터 저장소 로드 (allow_dangerous_deserialization=True 추가)
        vectorstore = FAISS.load_local(".", embeddings, allow_dangerous_deserialization=True)
        return True, vectorstore

    except Exception as e:
        logger.error(f"벡터 저장소 다운로드 오류: {e}")
        return False, str(e)

def get_conversation_chain(vectorstore, openai_api_key):
    """대화 체인 생성 함수"""
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4', temperature=0)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        ),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )

    return conversation_chain

def main():
    # 페이지 설정
    st.set_page_config(
        page_title="요리 도우미 V3",
        page_icon="🍳"
    )

    st.title("요리 도우미 V3 🍳")

    # 세션 상태 초기화
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "assistant", "content": "안녕하세요! 요리 도우미입니다. 어떤 요리에 대해 알고 싶으신가요?"}
        ]
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    # 사이드바 생성
    with st.sidebar:
        st.header("설정")
        
        # OpenAI API 키
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        if not openai_api_key:
            st.info("API 키를 입력해주세요.", icon="🔑")
            
        # 벡터 DB 로드 버튼
        load_button = st.button("벡터 DB 로드")

    # 벡터 DB 로드
    if load_button:
        if not validate_api_key(openai_api_key):
            st.error("유효한 OpenAI API 키를 입력해주세요.")
            st.stop()

        try:
            with st.spinner("벡터 DB를 불러오는 중..."):
                success, result = download_vector_store()
                
                if success:
                    st.session_state.vectorstore = result
                    st.session_state.conversation = get_conversation_chain(result, openai_api_key)
                    st.session_state.processComplete = True
                    st.success("벡터 DB를 성공적으로 불러왔습니다!")
                else:
                    st.error(f"벡터 DB 불러오기 실패: {result}")
                    
        except Exception as e:
            st.error(f"벡터 DB 불러오기 중 오류 발생: {e}")
            logger.error(f"벡터 DB 로드 오류: {e}")

    # 채팅 인터페이스
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # 사용자 입력 처리
    if query := st.chat_input("질문을 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.write(query)

        if not st.session_state.conversation:
            st.warning("먼저 벡터 DB를 불러와주세요.")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "죄송합니다. 먼저 벡터 DB를 불러와주세요."
            })
            st.rerun()

        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하는 중..."):
                try:
                    result = st.session_state.conversation({"question": query})
                    response = result['answer']
                    source_documents = result.get('source_documents', [])

                    st.write(response)

                    if source_documents:
                        with st.expander("참고 문서"):
                            for i, doc in enumerate(source_documents[:3], 1):
                                st.markdown(f"**참고 {i}:** {doc.metadata.get('source', '알 수 없는 출처')}")
                                st.markdown(f"```\n{doc.page_content[:200]}...\n```")

                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_message = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })
                    logger.error(f"응답 생성 오류: {e}")

if __name__ == '__main__':
    main()
