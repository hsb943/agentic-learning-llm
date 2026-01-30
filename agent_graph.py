import operator
import uuid
from typing import Annotated, List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utils.math import cosine_similarity


# =========================
# LLM + Tool Setup
# =========================

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
search_tool = TavilySearchResults(max_results=3)


# =========================
# Structured Output Models
# =========================

class LearningCheckpoint(BaseModel):
    description: str
    criteria: List[str]
    verification: str


class Checkpoints(BaseModel):
    checkpoints: List[LearningCheckpoint]


class SearchQuery(BaseModel):
    search_queries: List[str]


class LearningVerification(BaseModel):
    understanding_level: float = Field(..., ge=0, le=1)
    feedback: str
    suggestions: List[str]
    context_alignment: bool


class FeynmanTeaching(BaseModel):
    simplified_explanation: str
    key_concepts: List[str]
    analogies: List[str]


# =========================
# Agent State
# =========================

class LearningState(TypedDict, total=False):
    topic: str
    context: str
    context_key: str
    context_chunks: Annotated[List[str], operator.add]
    checkpoints: Checkpoints
    search_queries: SearchQuery
    current_checkpoint: int
    current_question: str
    current_answer: str
    verification: LearningVerification
    teaching: FeynmanTeaching


# =========================
# Context Store
# =========================

class ContextStore:
    def __init__(self):
        self.store = InMemoryStore()

    def save(self, chunks: List[str], embeddings_list: List[List[float]], key=None):
        namespace = ("context",)
        key = key or str(uuid.uuid4())
        self.store.put(namespace, key, {
            "chunks": chunks,
            "embeddings": embeddings_list
        })
        return key

    def load(self, key: str):
        namespace = ("context",)
        return self.store.get(namespace, key).value


context_store = ContextStore()


# =========================
# Core Nodes
# =========================

def generate_checkpoints(state: LearningState):
    structured = llm.with_structured_output(Checkpoints)
    response = structured.invoke([
        SystemMessage(content="Generate 3 learning checkpoints progressing from foundation to mastery."),
        HumanMessage(content=f"Topic: {state['topic']}")
    ])
    return {"checkpoints": response, "current_checkpoint": 0}


def generate_queries(state: LearningState):
    structured = llm.with_structured_output(SearchQuery)
    response = structured.invoke([
        SystemMessage(content="Generate one search query per checkpoint."),
        HumanMessage(content=str(state["checkpoints"]))
    ])
    return {"search_queries": response}


def search_web(state: LearningState):
    docs = []
    for q in state["search_queries"].search_queries:
        results = search_tool.invoke(q)
        docs.extend([r["content"] for r in results])

    chunk_embeddings = embeddings.embed_documents(docs)
    key = context_store.save(docs, chunk_embeddings)
    return {"context_chunks": docs, "context_key": key}


def generate_question(state: LearningState):
    checkpoint = state["checkpoints"].checkpoints[state["current_checkpoint"]]
    response = llm.invoke([
        SystemMessage(content="Generate a verification question."),
        HumanMessage(content=str(checkpoint))
    ])
    return {"current_question": response.content}


def verify_answer(state: LearningState):
    checkpoint = state["checkpoints"].checkpoints[state["current_checkpoint"]]
    context = context_store.load(state["context_key"])

    query_embedding = embeddings.embed_query(checkpoint.verification)
    similarities = cosine_similarity([query_embedding], context["embeddings"])[0]
    top_indices = sorted(range(len(similarities)),
                         key=lambda i: similarities[i],
                         reverse=True)[:3]

    relevant_chunks = [context["chunks"][i] for i in top_indices]

    structured = llm.with_structured_output(LearningVerification)
    response = structured.invoke([
        SystemMessage(content="Evaluate the answer."),
        HumanMessage(content=f"""
        Question: {state['current_question']}
        Answer: {state['current_answer']}
        Criteria: {checkpoint.criteria}
        Context: {relevant_chunks}
        """)
    ])

    return {"verification": response}


def teach_concept(state: LearningState):
    structured = llm.with_structured_output(FeynmanTeaching)
    response = structured.invoke([
        SystemMessage(content="Explain using Feynman technique."),
        HumanMessage(content=str(state["verification"]))
    ])
    return {"teaching": response}


def next_checkpoint(state: LearningState):
    return {"current_checkpoint": state["current_checkpoint"] + 1}


# =========================
# Routing Logic
# =========================

def route_after_verification(state: LearningState):
    if state["verification"].understanding_level < 0.7:
        return "teach_concept"

    if state["current_checkpoint"] + 1 < len(state["checkpoints"].checkpoints):
        return "next_checkpoint"

    return END


# =========================
# Graph Construction
# =========================

builder = StateGraph(LearningState)
memory = MemorySaver()

builder.add_node("generate_checkpoints", generate_checkpoints)
builder.add_node("generate_queries", generate_queries)
builder.add_node("search_web", search_web)
builder.add_node("generate_question", generate_question)
builder.add_node("verify_answer", verify_answer)
builder.add_node("teach_concept", teach_concept)
builder.add_node("next_checkpoint", next_checkpoint)

builder.add_edge(START, "generate_checkpoints")
builder.add_edge("generate_checkpoints", "generate_queries")
builder.add_edge("generate_queries", "search_web")
builder.add_edge("search_web", "generate_question")
builder.add_edge("generate_question", "verify_answer")

builder.add_conditional_edges(
    "verify_answer",
    route_after_verification,
    {
        "teach_concept": "teach_concept",
        "next_checkpoint": "next_checkpoint",
        END: END,
    },
)

builder.add_edge("teach_concept", "next_checkpoint")
builder.add_edge("next_checkpoint", "generate_question")

graph = builder.compile(checkpointer=memory)

