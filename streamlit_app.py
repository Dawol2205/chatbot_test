import streamlit as st
import tiktoken
import json
from loguru import logger
from concurrent import futures

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import JSONLoader
from langchain.document_loaders.json import JsonOutputParser

def validate_api_key(api_key):
    """OpenAI API 키 형식 검증"""
    return api_key and len(api_key) > 20

def process_json(file_path):
    """JSON 파일 처리 함수"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        # JSON을 텍스트로 변환
        processed_text = []
        
        def extract_text(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    if isinstance(value, (dict, list)):
                        extract_text(value, new_path)
                    else:
                        processed_text.append(f"{new_path}: {value}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_path = f"{path}[{i}]"
                    if isinstance(item, (dict, list)):
                        extract_text(item, new_path)
                    else:
                        processed_text.append(f"{new_path}: {item}")
                        
        extract_text(data)
        return "\n".join(processed_text)
    except Exception as e:
        logger.error(f"JSON 처리 중 오류 발생: {e}")
        raise e

def main():
    # 페이지 설정: 제목과 아이콘 지정
    st.set_page_config(
        page_title="요리 도우미",
        page_icon="🍳"
    )

    # 앱 제목 설정
    st.title("요리 도우미 🍳")

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

    # 사이드바 생성
    with st.sidebar:
        # 문서 업로드 기능
        uploaded_files = st.file_uploader(
            "요리 관련 문서 업로드", 
            type=["pdf", "docx", "pptx", "json"],  # JSON 추가
            accept_multiple_files=True
        )
        
        # OpenAI API 키 입력 필드
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        if not openai_api_key:
            st.info("API 키를 입력해주세요.", icon="🔑")
        
        # 문서 처리 버튼
        process_button = st.button("문서 처리")

    # 문서 처리 로직
    if process_button:
        if not validate_api_key(openai_api_key):
            st.error("유효한 API 키를 입력해주세요.")
            st.stop()
        
        if not uploaded_files:
            st.warning("처리할 문서를 업로드해주세요.")
            st.stop()

        try:
            with st.spinner("문서를 처리하는 중..."):
                # 문서에서 텍스트 추출
                docs = get_text(uploaded_files)
                
                if not docs:
                    st.error("처리할 수 있는 문서가 없습니다.")
                    st.stop()
                
                # 텍스트 청크 생성
                chunks = get_text_chunks(docs)
                
                # 임베딩 생성
                embeddings = HuggingFaceEmbeddings(
                    model_name="jhgan/ko-sroberta-multitask",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                # 벡터 저장소 생성
                vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
                
                # 대화 체인 초기화
                st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
                st.session_state.processComplete = True
                st.success("문서 처리 완료!")

        except Exception as e:
            st.error(f"문서 처리 중 오류 발생: {e}")
            logger.error(f"문서 처리 오류: {e}")

    # 채팅 인터페이스
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # 사용자 입력 처리
    if query := st.chat_input("질문을 입력하세요"):
        # 사용자 메시지를 대화 히스토리에 추가
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.write(query)

        # 대화 체인 준비 상태 확인
        if not st.session_state.conversation:
            st.warning("먼저 문서를 처리해주세요.")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "죄송합니다. 먼저 요리 관련 문서를 업로드하고 처리해주세요."
            })
            st.rerun()

        # 질의 처리
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

                    # 어시스턴트 응답을 대화 히스토리에 추가
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_message = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })
                    logger.error(f"응답 생성 오류: {e}")

def get_text(docs):
    """업로드된 문서에서 텍스트 추출하는 함수"""
    doc_list = []
    
    for doc in docs:
        try:
            # 문서 저장
            file_name = doc.name
            with open(file_name, "wb") as file:
                file.write(doc.getvalue())
                logger.info(f"업로드된 파일: {file_name}")

            # 파일 유형에 따라 로더 선택
            if '.pdf' in doc.name.lower():
                loader = PyPDFLoader(file_name)
                documents = loader.load_and_split()
            elif '.docx' in doc.name.lower():
                loader = Docx2txtLoader(file_name)
                documents = loader.load_and_split()
            elif '.pptx' in doc.name.lower():
                loader = UnstructuredPowerPointLoader(file_name)
                documents = loader.load_and_split()
            elif '.json' in doc.name.lower():
                # JSON 파일 처리
                text_content = process_json(file_name)
                documents = [Document(page_content=text_content, metadata={"source": file_name})]
            else:
                continue

            doc_list.extend(documents)
        except Exception as e:
            logger.error(f"문서 처리 중 오류 발생: {file_name}, 오류: {e}")
            continue
            
    return doc_list

# 나머지 함수들은 이전과 동일...
[이전 코드의 나머지 함수들: tiktoken_len, get_text_chunks, get_conversation_chain]

if __name__ == '__main__':
    main()
