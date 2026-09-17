from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from main import create_rag
# -------------------------
# LIFESPAN
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting API...")
    app.state.rag_chain = None
    print("API ready!")
    yield
    print("Shutting down...")
app = FastAPI(
    lifespan=lifespan
)
# -------------------------
# REQUEST MODEL
# -------------------------
class ChatRequest(BaseModel):
    question: str
    history: list[dict] = Field(default_factory=list)
# -------------------------
# RESPONSE MODEL
# -------------------------
class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[int]
# -------------------------
# HOME
# -------------------------
@app.get("/")
def home():
    return {
        "message": "Conversational RAG API is running"
    }
# -------------------------
# UPLOAD PDF
# -------------------------
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):
    # Check file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )
    # Save uploaded PDF
    pdf_path = f"uploaded_{file.filename}"
    with open(pdf_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    # Create RAG
    try:
        app.state.rag_chain = create_rag(
            pdf_path
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}"
        )
    return {
        "message": "PDF uploaded successfully",
        "filename": file.filename
    }
# -------------------------
# CHAT
# -------------------------
@app.post(
    "/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest):
    # Validate question
    if request.question.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="Question can't be empty"
        )
    # Check whether PDF was uploaded
    if app.state.rag_chain is None:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF first"
        )
    # Get RAG chain
    rag_chain = app.state.rag_chain
    # Convert history into LangChain messages
    from langchain_core.messages import (
        HumanMessage,
        AIMessage
    )
    chat_history = []

    for message in request.history:

        role = message.get("role")
        content = message.get("content")

        if not role or not content:
            continue

        if role == "user":
            chat_history.append(
                HumanMessage(content=content)
            )

        elif role == "assistant":
            chat_history.append(
                AIMessage(content=content)
            )
    # -------------------------
    # INVOKE RAG
    # -------------------------
    response = rag_chain.invoke(
        {
            "input": request.question,
            "chat_history": chat_history
        }
    )
    # -------------------------
    # GET SOURCES
    # -------------------------
    pages = [
        doc.metadata["page"] + 1
        for doc in response["context"]
    ]
    # Remove duplicate pages
    pages = list(dict.fromkeys(pages))
    # -------------------------
    # RETURN
    # -------------------------
    return {
        "question": request.question,
        "answer": response["answer"],
        "sources": pages
    }