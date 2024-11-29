import streamlit as st
import logging
import pickle
import base64
import requests
from github import Github
from datetime import datetime

from langchain_community.chat_models import ChatOpenAI
from langchain_community.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GitHub 저장소 정보
GITHUB_REPO = "Dawol2205/chatbot_test"
GITHUB_BRANCH = "main"
VECTOR_PATH = "vector_store"  # 벡터 저장소가 있는 디렉토리

def validate_api_key(api_key):
    """OpenAI API 키 형식 검증"""
    return api_key and len(api_key) > 20

def get_github_file_content(token, repo_name, file_path, branch="main"):
    """GitHub에서 파일 내용 가져오기"""
    try:
        g = Github(token)
        repo = g.get_repo(repo_name)
        content = repo.get_contents(file_path, ref=branch)
        
        decoded_content = base64.b64decode(content.content)
        return True, decoded_content
    except Exception as e:
        logger.error(f"GitHub 파일 로드 오류: {e}")
        return False, str(e)

def load_vector_store(github_token, filepath):
    """GitHub에서 벡터 저장소 로드"""
    try:
        success, content = get_github_file_content(github_token, GITHUB_REPO, filepath, GITHUB_BRANCH)
        if success:
            vectorstore = pickle.loads(content)
            return True, vectorstore
        else:
            return False, content
    except Exception as e:
        logger.error(f"벡터 저장소 로드 오류: {e}")
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

        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.experimental_rerun()

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

            # GitHub 토큰 입력
            github_token = st.text_input("GitHub Token", type="password")
            if not github_token:
                st.info("GitHub 토큰을 입력해주세요.", icon="🔑")

            # 벡터 DB 로드 버튼
            if st.button("벡터 DB 로드"):
                if not validate_api_key(openai_api_key):
                    st.error("유효한 OpenAI API 키를 입력해주세요.")
                    st.stop()

                try:
                    with st.spinner("벡터 DB를 불러오는 중..."):
                        # 벡터 저장소 불러오기
                        success, result = load_vector_store(github_token, f"{VECTOR_PATH}/index.pkl")
                        
                        if success:
                            st.session_state.vectorstore = result
                            st.session_state.conversation = get_conversation_chain(result, openai_api_key)
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
                st.experimental_rerun()

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
