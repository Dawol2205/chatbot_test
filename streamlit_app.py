import os
import json
import logging
import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='dirchat.log'
)

def main():
    st.set_page_config(
        page_title="DirChat",
        page_icon=":books:"
    )

    st.title("_Private Data :red[QA Chat]_ :books:")

    # 세션 상태 초기화
    session_keys = [
        "conversation", 
        "chat_history", 
        "processComplete", 
        "messages"
    ]
    for key in session_keys:
        if key not in st.session_state:
            st.session_state[key] = None

    with st.sidebar:
        # 파일 업로드
        uploaded_files = st.file_uploader(
            "파일 업로드", 
            type=['pdf', 'docx', 'pptx', 'json'],
            accept_multiple_files=True
        )

        # 업로드된 파일 초기화 및 검증
        if not uploaded_files:  # 파일이 업로드되지 않은 경우
            uploaded_files = []  # 빈 리스트로 초기화

        if len(uploaded_files) > 5:  # 파일 개수 제한
            st.warning("최대 5개의 파일만 업로드할 수 있습니다.")
            uploaded_files = uploaded_files[:5]

        # 10MB 파일 크기 제한
        filtered_files = []
        for file in uploaded_files:
            if file.size <= 10 * 1024 * 1024:  # 10MB
                filtered_files.append(file)
            else:
                st.warning(f"{file.name}은 10MB를 초과합니다. 건너뜁니다.")

        uploaded_files = filtered_files  # 유효 파일만 유지
        
        # 업로드 안내 표시
        st.sidebar.info("""
        📚 DirChat 사용 가이드
        - PDF, DOCX, PPTX, JSON 파일 지원
        - 최대 5개 파일 업로드 가능
        - 각 파일 10MB 제한
        """)

        openai_api_key = st.text_input("OpenAI API Key", type="password")
        process = st.button("문서 처리")

    if process:
        if not openai_api_key:
            st.info("OpenAI API 키를 입력해주세요.")
            st.stop()

        # 문서 처리
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vector_store = get_vectorstore(text_chunks)
     
        st.session_state.conversation = get_conversation_chain(vector_store, openai_api_key)
        st.session_state.processComplete = True

    # 초기 메시지 설정
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "assistant", 
             "content": "안녕하세요! 업로드된 문서에 대해 질문해주세요."}
        ]

    # 메시지 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # 채팅 입력 처리
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("답변 생성 중..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                
                # 참고 문서 확장 섹션
                with st.expander("참고 문서 확인"):
                    for doc in source_documents[:3]:
                        st.markdown(doc.metadata['source'], help=doc.page_content)

        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    doc_list = []
    supported_extensions = ['.pdf', '.docx', '.pptx', '.json']
    
    for doc in docs:
        file_extension = os.path.splitext(doc.name)[1].lower()
        if file_extension not in supported_extensions:
            st.warning(f"지원되지 않는 파일 형식: {doc.name}")
            continue
        
        try:
            with open(doc.name, "wb") as file:
                file.write(doc.getvalue())
                logger.info(f"파일 업로드: {doc.name}")
            
            if file_extension == '.pdf':
                loader = PyPDFLoader(doc.name)
                documents = loader.load_and_split()
            elif file_extension == '.docx':
                loader = Docx2txtLoader(doc.name)
                documents = loader.load_and_split()
            elif file_extension == '.pptx':
                loader = UnstructuredPowerPointLoader(doc.name)
                documents = loader.load_and_split()
            elif file_extension == '.json':
                with open(doc.name, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    
                    documents = []
                    if isinstance(json_data, list):
                        for item in json_data:
                            doc = Document(
                                page_content=str(item),
                                metadata={'source': doc.name}
                            )
                            documents.append(doc)
                    elif isinstance(json_data, dict):
                        doc = Document(
                            page_content=json.dumps(json_data, ensure_ascii=False),
                            metadata={'source': doc.name}
                        )
                        documents.append(doc)
            
            doc_list.extend(documents)
        except Exception as e:
            st.error(f"파일 처리 중 오류: {doc.name}, {e}")
    
    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    vectordb.save_local("faiss_index")  # 선택적: 인덱스 로컬 저장
    
    return vectordb

def get_conversation_chain(vetorestore, openai_api_key):
    try:
        llm = ChatOpenAI(
            openai_api_key=openai_api_key, 
            model_name='gpt-3.5-turbo',
            temperature=0,
            max_tokens=1000
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(
                search_type='mmr', 
                search_kwargs={'k': 3}
            ), 
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
    except Exception as e:
        st.error(f"대화 체인 생성 중 오류: {e}")
        return None

if __name__ == '__main__':
    main()
