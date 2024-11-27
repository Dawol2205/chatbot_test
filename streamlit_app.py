import secrets
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

import streamlit as st
from loguru import logger

def get_openai_callback():
    return {"api_key": secrets.token_urlsafe(16)}  # get_openai_callback() 메서드에 api_key을 추가

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

# 파일 처리 최적화
def get_text(docs):
    try:
        doc_list = []
        
        for doc in docs:
            file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
            
            with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
                file.write(doc.getvalue())
                logger.info(f"Uploaded {file_name}")
            
            loader = None
            if '.pdf' in doc.name:
                loader = PyPDFLoader(file_name)
            elif '.docx' in doc.name:
                loader = Docx2txtLoader(file_name)
            elif '.pptx' in doc.name:
                loader = UnstructuredPowerPointLoader(file_name)

            try:
                documents = loader.load_and_split()
            except Exception as e:
                logger.error(f"Failed to process file {file_name}: {e}")
            
            doc_list.extend(documents)
        return doc_list
    except Exception as e:
        logger.error(f"Failed to read files: {e}")
        return None

# 텍스트 쪼기 최적화
def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=100,
            length_function=tiktoken_len
        )
        chunks = text_splitter.split_documents(text)
        return chunks
    except Exception as e:
        logger.error(f"Failed to split documents: {e}")
        return None

# 벡터 스토어 최적화
def get_vectorstore(text_chunks):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )  
        
        vectordb = FAISS.from_documents(text_chunks, embeddings)
        return vectordb
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        return None

# 대화 chain 최적화
def get_conversation_chain(vetorestore, openai_api_key):
    try:
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0)
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type='mmr', vervose=True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )
        
        return conversation_chain
    except Exception as e:
        logger.error(f"Failed to create conversation chain: {e}")
        return None

def main():
    st.set_page_config(
        page_title="DirChat",
        page_icon=":books:"
    )

    st.title("_Private Data :red[QA Chat]_ :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        
        process = st.button("Process")

    if process:
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)

        try:
            st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key) 
            st.session_state.processComplete = True
        except Exception as e:
            logger.error(f"Failed to process files: {e}")

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    if query := st.chat_input("질문을 입력해주세요."):
        try:
            st.session_state.messages.append({"role": "user", "content": query})
            
            with st.chat_message("user"):
                st.markdown(query)
                
            with st.chat_message("assistant"):
                chain = st.session_state.conversation

                with st.spinner("Thinking..."):
                    result = chain({"question": query})
                    with get_openai_callback() as cb:
                        st.session_state.chat_history = result['chat_history']
                    response = result['answer']
                    source_documents = result['source_documents']

                    st.markdown(response)
                    with st.expander("참고 문서 확인"):
                        st.markdown(source_documents[0].metadata['source'], help=source_documents[0].page_content)
                        st.markdown(source_documents[1].metadata['source'], help=source_documents[1].page_content)
                        st.markdown(source_documents[2].metadata['source'], help=source_documents[2].page_content)

            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            logger.error(f"Failed to process query: {e}")

if __name__ == '__main__':
    main()
