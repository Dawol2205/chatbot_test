import streamlit as st
import tiktoken
import json
import os
import requests
from loguru import logger
import pickle
from datetime import datetime

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document

def validate_api_key(api_key):
    """OpenAI API 키 형식 검증"""
    return api_key and len(api_key) > 20

def save_vectorstore_local(vectorstore, directory="vectorstore"):
    """벡터 저장소를 로컬에 저장하는 함수"""
    try:
        # 저장 디렉토리가 없으면 생성
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # 파일명 생성 (타임스탬프 포함)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(directory, f"vectorstore_{timestamp}.pkl")
        
        # 벡터 저장소를 파일로 저장
        with open(file_path, 'wb') as f:
            pickle.dump(vectorstore, f)
        
        return True, file_path
        
    except Exception as e:
        logger.error(f"로컬 저장 오류: {e}")
        return False, str(e)

def load_vectorstore_local(file_path):
    """로컬에서 벡터 저장소를 불러오는 함수"""
    try:
        # 파일에서 벡터 저장소 불러오기
        with open(file_path, 'rb') as f:
            vectorstore = pickle.load(f)
        
        return True, vectorstore
        
    except Exception as e:
        logger.error(f"로컬 로드 오류: {e}")
        return False, str(e)

def process_json(file_path):
    """JSON 파일을 처리하는 함수"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # JSON 데이터를 문자열로 변환
        text_content = json.dumps(data, ensure_ascii=False, indent=2)
        
        # Document 객체 생성
        return [Document(
            page_content=text_content,
            metadata={"source": file_path}
        )]
    except Exception as e:
        logger.error(f"JSON 파일 처리 중 오류 발생: {e}")
        return []

def tiktoken_len(text):
    """텍스트의 토큰 길이 계산 함수"""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    """업로드된 문서에서 텍스트 추출하는 함수"""
    doc_list = []
    
    for doc in docs:
        try:
            # 문서가 비어있는지 확인
            if doc is None or doc.size == 0:
                logger.warning(f"빈 문서가 감지됨: {doc.name if doc else 'Unknown'}")
                continue

            # 문서 저장
            file_name = doc.name
            with open(file_name, "wb") as file:
                file.write(doc.getvalue())
                logger.info(f"업로드된 파일: {file_name}")

            documents = []
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
                documents = process_json(file_name)

            # 문서가 성공적으로 로드되었는지 확인
            if documents:
                doc_list.extend(documents)
            else:
                logger.warning(f"문서에서 텍스트를 추출할 수 없음: {file_name}")

            # 임시 파일 삭제
            try:
                os.remove(file_name)
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {file_name}, 오류: {e}")

        except Exception as e:
            logger.error(f"문서 처리 중 오류 발생: {getattr(doc, 'name', 'Unknown')}, 오류: {e}")
            continue
    
    # 최종 결과 확인
    if not doc_list:
        raise ValueError("처리할 수 있는 문서가 없습니다. 문서를 확인해주세요.")
            
    return doc_list

def get_text_chunks(text):
    """텍스트를 일정 크기의 청크로 분할하는 함수"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

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
        page_title="요리 도우미",
        page_icon="🍳"
    )

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
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    # 사이드바 생성
    with st.sidebar:
        st.header("설정")
        
        # OpenAI API 키
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        if not openai_api_key:
            st.info("API 키를 입력해주세요.", icon="🔑")
        
        # 문서 업로드 섹션
        st.header("문서 업로드")
        uploaded_files = st.file_uploader(
            "요리 관련 문서 업로드",
            type=["pdf", "docx", "pptx", "json"],
            accept_multiple_files=True
        )
        
        # 버튼들을 2열로 배치
        col1, col2 = st.columns(2)
        with col1:
            process_button = st.button("문서 처리")
        with col2:
            save_local_button = st.button("벡터 저장")
        
        # 로컬 파일 업로더 추가
        st.header("벡터 파일 불러오기")
        local_file = st.file_uploader("벡터 파일 선택", type=["pkl"])
        if local_file:
            load_local_button = st.button("벡터 불러오기")

    # 로컬 파일 불러오기 버튼 처리
    if local_file and load_local_button:
        if not validate_api_key(openai_api_key):
            st.error("유효한 OpenAI API 키를 입력해주세요.")
            st.stop()

        try:
            with st.spinner("벡터 저장소를 불러오는 중..."):
                # 임시 파일로 저장
                with open("temp_vector.pkl", "wb") as f:
                    f.write(local_file.getvalue())
                
                # 벡터 저장소 불러오기
                success, result = load_vectorstore_local("temp_vector.pkl")
                
                if success:
                    st.session_state.vectorstore = result
                    st.session_state.conversation = get_conversation_chain(result, openai_api_key)
                    st.session_state.processComplete = True
                    st.success("벡터 저장소를 성공적으로 불러왔습니다!")
                else:
                    st.error(f"벡터 저장소 불러오기 실패: {result}")
                    
        except Exception as e:
            st.error(f"벡터 파일 불러오기 중 오류 발생: {e}")
            logger.error(f"로컬 로드 오류: {e}")

    # 문서 처리 로직
    if process_button:
        if not validate_api_key(openai_api_key):
            st.error("유효한 OpenAI API 키를 입력해주세요.")
            st.stop()
        
        if not uploaded_files:
            st.warning("처리할 문서를 업로드해주세요.")
            st.stop()

        try:
            with st.spinner("문서를 처리하는 중..."):
                # 문서에서 텍스트 추출
                try:
                    docs = get_text(uploaded_files)
                    if not docs:
                        st.error("처리할 수 있는 내용이 문서에 없습니다.")
                        st.stop()
                except ValueError as e:
                    st.error(str(e))
                    st.stop()
                except Exception as e:
                    st.error(f"문서 처리 중 오류가 발생했습니다: {str(e)}")
                    st.stop()
                
                # 텍스트 청크 생성
                chunks = get_text_chunks(docs)
                if not chunks:
                    st.error("문서를 청크로 분할할 수 없습니다.")
                    st.stop()
                
                # 임베딩 생성
                embeddings = HuggingFaceEmbeddings(
                    model_name="jhgan/ko-sroberta-multitask",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                # 벡터 저장소 생성
                vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
                
                # 세션에 벡터 저장소 저장
                st.session_state.vectorstore = vectorstore
                
                # 대화 체인 초기화
                st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
                st.session_state.processComplete = True
                st.success("문서 처리 완료!")

        except Exception as e:
            st.error(f"문서 처리 중 오류 발생: {str(e)}")
            logger.error(f"문서 처리 오류: {e}")

    # 벡터 저장소 로컬 저장
    if save_local_button:
        if not st.session_state.vectorstore:
            st.error("저장할 벡터 데이터가 없습니다. 먼저 문서를 처리해주세요.")
            st.stop()
        
        try:
            with st.spinner("벡터 저장소를 저장하는 중..."):
                success, result = save_vectorstore_local(
                    st.session_state.vectorstore,
                    directory="vectorstore"
                )
                if success:
                    st.success(f"벡터 저장소를 저장했습니다! (경로: {result})")
                else:
                    st.error(f"벡터 저장소 저장 실패: {result}")
        except Exception as e:
            st.error(f"벡터 저장소 저장 중 오류 발생: {e}")
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
            st.warning("먼저 문서를 처리하거나 벡터 저장소를 불러와주세요.")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "죄송합니다. 먼저 요리 관련 문서를 업로드하거나 기존 벡터 저장소를 불러와주세요."
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
