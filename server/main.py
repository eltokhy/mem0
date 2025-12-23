import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from mem0 import Memory

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------
load_dotenv()

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "postgres")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
POSTGRES_COLLECTION_NAME = os.environ.get("POSTGRES_COLLECTION_NAME", "memories")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "mem0graph")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MEM0_API_KEY = os.environ.get("MEM0_API_KEY")
HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", "/app/history/history.db")

# ------------------------------------------------------------------------------
# Default Config
# ------------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "version": "v1.1",
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "host": POSTGRES_HOST,
            "port": int(POSTGRES_PORT),
            "dbname": POSTGRES_DB,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "collection_name": POSTGRES_COLLECTION_NAME,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": NEO4J_URI,
            "username": NEO4J_USERNAME,
            "password": NEO4J_PASSWORD,
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "api_key": OPENAI_API_KEY,
            "temperature": 0.2,
            "model": "gpt-4.1-nano-2025-04-14",
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": OPENAI_API_KEY,
            "model": "text-embedding-3-small",
        },
    },
    "history_db_path": HISTORY_DB_PATH,
}

MEMORY_INSTANCE = Memory.from_config(DEFAULT_CONFIG)

# ------------------------------------------------------------------------------
# App
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Mem0 REST APIs",
    description="A REST API for managing and searching memories for AI agents and apps.",
    version="1.0.0",
)

# ------------------------------------------------------------------------------
# CORS Middleware
# ------------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dashboard.mem0.enginecy.cloud",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Security
# ------------------------------------------------------------------------------
security = HTTPBearer()

def verify_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    # Allow requests from dashboard IP without API key
    client_ip = request.client.host
    if client_ip in ["10.0.1.151", "172.23.0.5"]:  # Dashboard container IPs
        return None
    
    # For all other requests, require API key
    if not MEM0_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")
    if not credentials or credentials.credentials != MEM0_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return credentials.credentials

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str


class MemoryCreate(BaseModel):
    messages: List[Message]
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MemoryFilter(BaseModel):
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def safe_dump(model):
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()

# ------------------------------------------------------------------------------
# API Router (all routes under /api/v1/)
# ------------------------------------------------------------------------------
api_router = APIRouter(prefix="/api/v1")

@api_router.post("/configure", summary="Configure Mem0")
def set_config(
    request: Request,
    config: Dict[str, Any],
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    verify_api_key(request, credentials)
    global MEMORY_INSTANCE
    MEMORY_INSTANCE = Memory.from_config(config)
    return {"message": "Configuration set successfully"}


@api_router.post("/memories/", summary="Create memories (with trailing slash)")
@api_router.post("/memories", summary="Create memories")
def add_memory(
    request: Request,
    memory_create: MemoryCreate,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    verify_api_key(request, credentials)

    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(
            status_code=400,
            detail="At least one identifier (user_id, agent_id, run_id) is required.",
        )

    data = safe_dump(memory_create)
    params = {k: v for k, v in data.items() if v is not None and k != "messages"}

    try:
        response = MEMORY_INSTANCE.add(
            messages=[safe_dump(m) for m in memory_create.messages],
            **params,
        )
        return JSONResponse(content=response)
    except Exception as e:
        logging.exception("Error in add_memory")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/memories/filter/", summary="Filter memories (POST with trailing slash)")
@api_router.post("/memories/filter", summary="Filter memories (POST)")
async def filter_memories_post(
    request: Request,
    filter_req: Optional[MemoryFilter] = None,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    verify_api_key(request, credentials)
    
    if filter_req:
        user_id = filter_req.user_id
        agent_id = filter_req.agent_id
        run_id = filter_req.run_id
    else:
        user_id = agent_id = run_id = None
    
    if not any([user_id, agent_id, run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    
    try:
        params = {k: v for k, v in {
            "user_id": user_id,
            "run_id": run_id,
            "agent_id": agent_id,
        }.items() if v is not None}
        
        return MEMORY_INSTANCE.get_all(**params)
    except Exception as e:
        logging.exception("Error in filter_memories_post")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/memories/filter/", summary="Get memories with filters (trailing slash)")
@api_router.get("/memories/filter", summary="Get memories with filters")
@api_router.get("/memories/", summary="Get all memories (trailing slash)")
@api_router.get("/memories", summary="Get all memories")
def get_all_memories(
    request: Request,
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    verify_api_key(request, credentials)

    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")

    try:
        params = {k: v for k, v in {
            "user_id": user_id,
            "run_id": run_id,
            "agent_id": agent_id,
        }.items() if v is not None}

        return MEMORY_INSTANCE.get_all(**params)
    except Exception as e:
        logging.exception("Error in get_all_memories")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/memories/{memory_id}", summary="Get a memory")
def get_memory(
    request: Request,
    memory_id: str,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    verify_api_key(request, credentials)
    try:
        return MEMORY_INSTANCE.get(memory_id)
    except Exception as e:
        logging.exception("Error in get_memory")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/search", summary="Search memories")
def search_memories(
    request: Request,
    search_req: SearchRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    verify_api_key(request, credentials)
    try:
        params = {k: v for k, v in safe_dump(search_req).items() if v is not None and k != "query"}
        return MEMORY_INSTANCE.search(query=search_req.query, **params)
    except Exception as e:
        logging.exception("Error in search_memories")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.put("/memories/{memory_id}", summary="Update a memory")
def update_memory(
    request: Request,
    memory_id: str,
    updated_memory: Dict[str, Any],
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    verify_api_key(request, credentials)
    try:
        return MEMORY_INSTANCE.update(memory_id=memory_id, data=updated_memory)
    except Exception as e:
        logging.exception("Error in update_memory")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/memories/{memory_id}/history", summary="Get memory history")
def memory_history(
    request: Request,
    memory_id: str,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    verify_api_key(request, credentials)
    try:
        return MEMORY_INSTANCE.history(memory_id=memory_id)
    except Exception as e:
        logging.exception("Error in memory_history")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.delete("/memories/{memory_id}", summary="Delete a memory")
def delete_memory(
    request: Request,
    memory_id: str,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    verify_api_key(request, credentials)
    try:
        MEMORY_INSTANCE.delete(memory_id=memory_id)
        return {"message": "Memory deleted successfully"}
    except Exception as e:
        logging.exception("Error in delete_memory")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.delete("/memories", summary="Delete all memories")
def delete_all_memories(
    request: Request,
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    verify_api_key(request, credentials)

    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")

    try:
        params = {k: v for k, v in {
            "user_id": user_id,
            "run_id": run_id,
            "agent_id": agent_id,
        }.items() if v is not None}

        MEMORY_INSTANCE.delete_all(**params)
        return {"message": "All relevant memories deleted"}
    except Exception as e:
        logging.exception("Error in delete_all_memories")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/reset", summary="Reset all memories")
def reset_memory(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    verify_api_key(request, credentials)
    try:
        MEMORY_INSTANCE.reset()
        return {"message": "All memories reset"}
    except Exception as e:
        logging.exception("Error in reset_memory")
        raise HTTPException(status_code=500, detail=str(e))


# Include the API router
app.include_router(api_router)

# ------------------------------------------------------------------------------
# Root redirect
# ------------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")
