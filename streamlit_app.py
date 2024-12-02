import streamlit as st
import logging
import pickle
import json
import os
from datetime import datetime
from gtts import gTTS
import base64
import tempfile
import requests

from langchain.chat_models import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """세션 상태 초기화"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "안녕하세요! 요리 도우미입니다. 어떤 요리에 대해 알고 싶으신가요?",
                "audio": None
            }
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

def autoplay_audio(audio_content, autoplay=True):
    """음성 자동 재생을 위한 HTML 컴포넌트 생성"""
    b64 = base64.b64encode(audio_content).decode()
    autoplay_attr = 'autoplay' if autoplay else ''
    md = f"""
        <audio controls {autoplay_attr}>
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
        os.remove(fp.name)
        return audio_bytes
    except Exception as e:
        logger.error(f"음성 변환 오류: {e}")
        return None

def load_vector_database():
    """GitHub 저장소에서 JSON 파일들을 처리하고 벡터 데이터베이스를 로드"""
    repo_path = "Dawol2205/chatbot_test"
    folder_path = "food_DB"
    api_url = f"https://api.github.com/repos/{repo_path}/contents/{folder_path}"

    response = requests.get(api_url)
    files = response.json()

    documents = []
    for file in files:
        if file['name'].endswith('.json'):
            content_response = requests.get(file['download_url'])
            content = content_response.json()
            metadata = {'source': file['name']}
            doc = Document(page_content=json.dumps(content, ensure_ascii=False), metadata=metadata)
            documents.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')

    if os.path.exists("index.pkl") and os.path.exists("index.faiss"):
        with open("index.pkl", "rb") as f:
            vector_store = pickle.load(f)
    else:
        vector_store = FAISS.from_documents(texts, embeddings)
        with open("index.pkl", "wb") as f:
            pickle.dump(vector_store, f)
        vector_store.save_local("index.faiss")

    return vector_store

def validate_api_key(api_key):
    """OpenAI API 키 형식 검증"""
    return api_key and len(api_key) > 20

def get_conversation_chain(vectorstore, openai_api_key, custom_prompt):
    """대화 체인 생성"""
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4', temperature=0)
    
    PROMPT = PromptTemplate(
        template=custom_prompt,
        input_variables=["context", "question"]
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        ),
        combine_docs_chain_kwargs={"prompt": PROMPT},
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
            
            # API 키 입력
            openai_api_key = st.text_input("OpenAI API Key", type="password")
            if not openai_api_key:
                st.info("OpenAI API 키를 입력해주세요.", icon="🔑")

            # 프롬프트 템플릿 설정
            st.header("프롬프트 템플릿")
            custom_prompt = st.text_area("RAG 프롬프트", value=st.session_state.custom_prompt)
            if custom_prompt != st.session_state.custom_prompt:
                st.session_state.custom_prompt = custom_prompt

            # GitHub 파일 처리 섹션
            st.header("데이터 로드")
            if st.button("요리 데이터 가져오기"):
                if not validate_api_key(openai_api_key):
                    st.error("유효한 OpenAI API 키를 입력해주세요.")
                    st.stop()

                try:
                    with st.spinner("데이터를 처리하는 중..."):
                        vector_store = load_vector_database()
                        
                        # 세션에 저장
                        st.session_state.vectorstore = vector_store
                        st.session_state.conversation = get_conversation_chain(
                            vector_store, 
                            openai_api_key,
                            st.session_state.custom_prompt
                        )
                        st.success("데이터 로드 완료!")
                except Exception as e:
                    st.error(f"처리 중 오류 발생: {str(e)}")
                    logger.error(f"처리 오류: {e}")

        # 채팅 인터페이스
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    # 어시스턴트 메시지에 대해 음성 컨트롤 추가
                    if message["role"] == "assistant" and st.session_state.voice_enabled:
                        if message.get("audio") is None and message["content"]:
                            # 음성이 아직 생성되지 않은 경우 생성
                            audio_bytes = text_to_speech(message["content"])
                            if audio_bytes:
                                message["audio"] = audio_bytes

                        if message.get("audio"):
                            # 음성 컨트롤 표시
                            cols = st.columns([1, 4])
                            with cols[0]:
                                if st.button("🔊 재생", key=f"play_message_{message['content']}"):
                                    autoplay_audio(message["audio"])
                            with cols[1]:
                                # 오디오 플레이어 표시 (컨트롤 포함)
                                autoplay_audio(message["audio"], autoplay=False)

        # 사용자 입력 처리
        if query := st.chat_input("질문을 입력하세요"):
            st.session_state.messages.append({"role": "user", "content": query, "audio": None})
            
            with st.chat_message("user"):
                st.write(query)

            if not st.session_state.conversation:
                response = "죄송합니다. 먼저 요리 데이터를 가져와주세요."
                st.warning(response)
                
                if st.session_state.voice_enabled:
                    audio_bytes = text_to_speech(response)
                else:
                    audio_bytes = None
                    
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "audio": audio_bytes
                })
                
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
                                message = st.session_state.messages[-1]
                                message["audio"] = audio_bytes
                                autoplay_audio(audio_bytes, autoplay=False)
                        else:
                            audio_bytes = None

                        if source_documents:
                            with st.expander("참고 문서"):
                                for i, doc in enumerate(source_documents[:3], 1):
                                    st.markdown(f"**참고 {i}:** {doc.metadata.get('source', '알 수 없는 출처')}")
                                    st.markdown(f"```\n{doc.page_content[:200]}...\n```")

                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "audio": audio_bytes
                        })

                    except Exception as e:
                        error_message = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
                        st.error(error_message)
                        
                        if st.session_state.voice_enabled:
                            audio_bytes = text_to_speech(error_message)
                        else:
                            audio_bytes = None
                            
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_message,
                            "audio": audio_bytes
                        })
                        
                        if audio_bytes:
                            autoplay_audio(audio_bytes)
                        
                        logger.error(f"응답 생성 오류: {e}")

    except Exception as e:
        logger.error(f"앱 실행 중 오류 발생: {e}")
        st.error("앱 실행 중 오류가 발생했습니다. 새로고침을 시도해주세요.")

if __name__ == '__main__':
    main()
