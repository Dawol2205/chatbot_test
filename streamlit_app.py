import streamlit as st
import logging
import pickle
import base64
import requests
import json
import os
from github import Github
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_api_key(api_key):
    """OpenAI API 키 형식 검증"""
    return api_key and len(api_key) > 20

def process_json(file):
    """JSON 파일을 처리하는 함수"""
    try:
        content = file.getvalue().decode('utf-8')
        data = json.loads(content)
        
        # JSON 데이터를 문자열로 변환
        text_content = json.dumps(data, ensure_ascii=False, indent=2)
        
        # Document 객체 생성
        return [Document(
            page_content=text_content,
            metadata={"source": file.name}
        )]
    except Exception as e:
        logger.error(f"JSON 파일 처리 중 오류 발생: {e}")
        return []

def get_text_chunks(documents):
    """텍스트를 청크로 분할"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100
    )
    return text_splitter.split_documents(documents)

def create_vector_store(documents):
    """벡터 저장소 생성"""
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    return FAISS.from_documents(documents=documents, embedding=embeddings)

def save_vector_store(vectorstore, directory="vector_store"):
    """벡터 저장소를 로컬에 저장"""
    try:
        # 저장 디렉토리가 없으면 생성
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        file_path = os.path.join(directory, "index.pkl")
        
        # 벡터 저장소를 파일로 저장
        with open(file_path, 'wb') as f:
            pickle.dump(vectorstore, f)
        
        return True, file_path
    except Exception as e:
        logger.error(f"저장 오류: {e}")
        return False, str(e)

def get_conversation_chain(vectorstore, openai_api_key):
    """대화 체인 생성"""
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
    try:
        # 페이지 설정
        st.set_page_config(
            page_title="요리 도우미",
            page_icon="🍳",
            layout="wide"
        )

        st.title("요리 도우미 🍳")

        # 세션 상태 초기화
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if 'messages' not in st.session_state:
            st.session_state['messages'] = [
                {"role": "assistant", "content": "안녕하세요! 요리 도우미입니다. 어떤 요리에 대해 알고 싶으신가요?"}
            ]
        if 'vectorstore' not in st.session_state:
            st.session_state.vectorstore = None

        # 사이드바 설정
        with st.sidebar:
            st.header("설정")
            
            # API 키 입력
            openai_api_key = st.text_input("OpenAI API Key", type="password")
            if not openai_api_key:
                st.info("OpenAI API 키를 입력해주세요.", icon="🔑")

            # JSON 파일 업로드
            st.header("JSON 파일 업로드")
            uploaded_file = st.file_uploader("JSON 파일 선택", type=["json"])
            
            # 처리 버튼들
            col1, col2 = st.columns(2)
            with col1:
                process_button = st.button("파일 처리")
            with col2:
                save_button = st.button("벡터 저장")

        # JSON 파일 처리
        if uploaded_file and process_button:
            if not validate_api_key(openai_api_key):
                st.error("유효한 OpenAI API 키를 입력해주세요.")
                st.stop()

            try:
                with st.spinner("JSON 파일 처리 중..."):
                    # JSON 처리
                    documents = process_json(uploaded_file)
                    if not documents:
                        st.error("JSON 파일 처리에 실패했습니다.")
                        st.stop()
                    
                    # 청크 생성
                    chunks = get_text_chunks(documents)
                    
                    # 벡터 저장소 생성
                    vectorstore = create_vector_store(chunks)
                    
                    # 세션에 저장
                    st.session_state.vectorstore = vectorstore
                    st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
                    st.success("JSON 파일 처리 완료!")

            except Exception as e:
                st.error(f"파일 처리 중 오류 발생: {str(e)}")
                logger.error(f"처리 오류: {e}")

        # 벡터 저장소 저장
        if save_button:
            if not st.session_state.vectorstore:
                st.error("저장할 벡터 데이터가 없습니다. 먼저 JSON 파일을 처리해주세요.")
                st.stop()

            try:
                with st.spinner("벡터 저장소 저장 중..."):
                    success, result = save_vector_store(st.session_state.vectorstore)
                    if success:
                        st.success(f"벡터 저장소를 저장했습니다! (경로: {result})")
                    else:
                        st.error(f"저장 실패: {result}")

            except Exception as e:
                st.error(f"저장 중 오류 발생: {str(e)}")
                logger.error(f"저장 오류: {e}")

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
                st.warning("먼저 JSON 파일을 처리해주세요.")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "죄송합니다. 먼저 JSON 파일을 업로드하고 처리해주세요."
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

    except Exception as e:
        logger.error(f"앱 실행 중 오류 발생: {e}")
        st.error("앱 실행 중 오류가 발생했습니다. 새로고침을 시도해주세요.")

if __name__ == '__main__':
    main()
