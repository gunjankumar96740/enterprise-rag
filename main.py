from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
import os
from dotenv import load_dotenv
def create_rag(pdf_path):
    # -------------------------
    # ENVIRONMENT VARIABLES
    # -------------------------
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found")
    # -------------------------
    # 1. PDF LOADING
    # -------------------------
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print("Total pages:", len(docs))
    # -------------------------
    # 2. CHUNKING
    # -------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    print("Total chunks:", len(chunks))
    # -------------------------
    # 3. EMBEDDINGS
    # -------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # -------------------------
    # 4. FAISS
    # -------------------------
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )
    print("FAISS index created.")
    # -------------------------
    # 5. RETRIEVER
    # -------------------------
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 10
        }
    )
    # -------------------------
    # 6. LLM
    # -------------------------
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key
    )
    # ==================================================
    # CONVERSATIONAL RAG
    # ==================================================
    # -------------------------
    # 7. HISTORY-AWARE PROMPT
    # -------------------------
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Given the chat history and the latest user question,
rewrite the question as a standalone question that can be
understood without the chat history.
Do NOT answer the question.
Only rewrite it if necessary."""
        ),
        MessagesPlaceholder(
            "chat_history"
        ),
        (
            "human",
            "{input}"
        )
    ])
    # -------------------------
    # 8. HISTORY-AWARE RETRIEVER
    # -------------------------
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )
    # -------------------------
    # 9. ANSWER PROMPT
    # -------------------------
    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful assistant.
Answer the user's question using ONLY the provided context.
If the answer is not present in the context, say:
"I don't know based on the provided document."
Context:
{context}"""
        ),

        MessagesPlaceholder(
            "chat_history"
        ),
        (
            "human",
            "{input}"
        )
    ])
    # -------------------------
    # 10. DOCUMENT CHAIN
    # -------------------------
    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt
    )
    # -------------------------
    # 11. FINAL RAG CHAIN
    # -------------------------
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )
    print("Conversational RAG ready!")
    return rag_chain