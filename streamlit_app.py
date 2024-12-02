import streamlit as st
import logging
import pickle
import json
import os
from datetime import datetime
from gtts import gTTS
import base64
import tempfile

from langchain_openai import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 벡터 저장소 경로
VECTOR_PATH = "vectorstore"

def autoplay_audio(audio_content):
    """음성 자동 재생을 위한 HTML 컴포넌트 생성"""
    b64 = base64.b64encode(audio_content).decode()
    md = f"""
        <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)

def text_to_speech(text, lang='ko'):
    """텍스트를 음성으로 변환"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts = gTTS(text=text, lang=lang)
            tts.save(fp.name)
            with open(fp.name, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            os.unlink(fp.name)
            return audio_bytes
    except Exception as e:
        logger.error(f"음성 변환 오류: {e}")
        return None

def initialize_session_state():
    """세션 상태 초기화"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "안녕하세요! 요리 도우미입니다. 어떤 요리에 대해 알고 싶으신가요?"}
        ]
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "custom_prompt" not in st.session_state:
        st.session_state.custom_prompt = """
아래 정보를 기반으로 사용자의 질문에 답변해주세요:
{context}

사용자 질문: {question}
답변: 주어진 정보를 바탕으로 상세하게 답변하겠습니다.
"""
    if "voice_enabled" not in st.session_state:
        st.session_state.voice_enabled = True

[이전 코드의 나머지 함수들은 동일하게 유지...]

def main():
    try:
        # 페이지 설정
        st.set_page_config(
            page_title="요리 도우미",
            page_icon="🍳",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # 세션 상태 초기화
        initialize_session_state()

        st.title("요리 도우미 🍳")

        # 사이드바 설정
        with st.sidebar:
            st.header("설정")
            
            # 음성 출력 토글
            st.session_state.voice_enabled = st.toggle("음성 출력 활성화", value=st.session_state.voice_enabled)
            
            [이전 사이드바 코드는 동일하게 유지...]

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
                response = "죄송합니다. 먼저 JSON 파일을 업로드하고 처리하거나 저장된 벡터를 불러와주세요."
                st.warning(response)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                
                if st.session_state.voice_enabled:
                    audio_bytes = text_to_speech(response)
                    if audio_bytes:
                        autoplay_audio(audio_bytes)
                
                st.stop()

            with st.chat_message("assistant"):
                with st.spinner("답변을 생성하는 중..."):
                    try:
                        result = st.session_state.conversation({"question": query})
                        response = result['answer']
                        source_documents = result.get('source_documents', [])

                        st.write(response)

                        # 음성 출력 처리
                        if st.session_state.voice_enabled:
                            audio_bytes = text_to_speech(response)
                            if audio_bytes:
                                autoplay_audio(audio_bytes)

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
                        
                        if st.session_state.voice_enabled:
                            audio_bytes = text_to_speech(error_message)
                            if audio_bytes:
                                autoplay_audio(audio_bytes)
                                
                        logger.error(f"응답 생성 오류: {e}")

    except Exception as e:
        logger.error(f"앱 실행 중 오류 발생: {e}")
        st.error("앱 실행 중 오류가 발생했습니다. 새로고침을 시도해주세요.")

if __name__ == '__main__':
    main()
