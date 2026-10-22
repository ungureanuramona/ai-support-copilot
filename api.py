
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from copilot import (
    EMBEDDING_MODEL_NAME,
    find_relevant_documents,
    generate_support_plan,
    load_documents,
)


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"

embedding_model = None
model_lock = Lock()


class TicketRequest(BaseModel):
    title: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=10, max_length=10000)


class SourceResult(BaseModel):
    filename: str
    similarity: float


class AnalysisResponse(BaseModel):
    category: str
    support_plan: str
    sources: list[SourceResult]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model
    yield
    embedding_model = None


app = FastAPI(
    title="AI Support Copilot API",
    description="AI-assisted incident investigation API",
    version="1.0.0",
    lifespan=lifespan,
)


def get_embedding_model():
    global embedding_model

    if embedding_model is None:
        with model_lock:
            if embedding_model is None:
                embedding_model = SentenceTransformer(
                    EMBEDDING_MODEL_NAME
                )

    return embedding_model


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ai-support-copilot",
    }


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_ticket(ticket: TicketRequest):
    documents = load_documents(KNOWLEDGE_BASE_DIR)

    try:
        model = get_embedding_model()

        relevant_documents = find_relevant_documents(
            ticket_text=f"{ticket.title}\n{ticket.description}",
            documents=documents,
            model=model,
        )

        support_plan = generate_support_plan(
            title=ticket.title,
            description=ticket.description,
            relevant_documents=relevant_documents,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="The analysis service is temporarily unavailable.",
        ) from exc

    return AnalysisResponse(
        category=(
            "Knowledge base match"
            if relevant_documents
            else "Insufficient knowledge base coverage"
        ),
        support_plan=support_plan,
        sources=[
            SourceResult(
                filename=document["source"],
                similarity=round(document["score"], 3),
            )
            for document in relevant_documents
        ],
    )
