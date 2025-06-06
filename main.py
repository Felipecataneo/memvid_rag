# mcp_server_fixed.py
#
# Servidor Memvid compatível com MCP usando FastAPI para melhor performance no Render

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import base64
import uuid
import logging
import os
from pathlib import Path
from typing import Generator, Dict, Any
import uvicorn
from pydantic import BaseModel

# Importações seguras dos módulos Memvid
try:
    from memvid import MemvidEncoder, MemvidChat
    from memvid.config import get_codec_parameters

except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logging.error(f"Erro crítico ao importar Memvid: {e}. Verifique a instalação.")
    exit(1)

# --- CONFIGURAÇÃO ---
PORT = int(os.getenv("PORT", 8000))
HOST = "0.0.0.0"
OUTPUT_DIR = Path("session_memories")
VIDEO_CODEC = "mp4v"
LLM_PROVIDER = os.getenv("MEMVID_PROVIDER", "google")
VIDEO_EXT = get_codec_parameters(VIDEO_CODEC).get("video_file_type", "mp4")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
OUTPUT_DIR.mkdir(exist_ok=True)

# --- MODELOS PYDANTIC ---
class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any] = {}
    id: int | str = 1

class CreateMemoryRequest(BaseModel):
    filename: str
    file_content_base64: str

class ChatRequest(BaseModel):
    query: str
    memory_id: str

# --- FASTAPI APP ---
app = FastAPI(title="Memvid MCP Server", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ROTAS MCP ---
@app.post("/mcp")
async def handle_mcp_request(request: MCPRequest):
    """Endpoint principal para requisições MCP seguindo o padrão JSON-RPC"""
    try:
        if request.method == "tools/list":
            return handle_list_tools(request.id)
        elif request.method == "tools/call":
            return await handle_call_tool(request.params, request.id)
        else:
            return create_error_response(request.id, -32601, f"Método não encontrado: {request.method}")
    except Exception as e:
        logger.error(f"Erro no MCP: {e}", exc_info=True)
        return create_error_response(request.id, -32603, f"Erro interno: {str(e)}")

def handle_list_tools(request_id):
    """Lista as ferramentas disponíveis no formato MCP padrão"""
    tools = [
        {
            "name": "create_memory_from_file",
            "description": "Cria uma memória de conhecimento a partir de um arquivo (PDF, TXT, MD) e retorna um ID para uso posterior.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Nome do arquivo."},
                    "file_content_base64": {"type": "string", "description": "Conteúdo do arquivo em base64."}
                },
                "required": ["filename", "file_content_base64"]
            }
        },
        {
            "name": "chat_with_memory",
            "description": "Inicia um chat com uma memória específica usando streaming.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A pergunta do usuário."},
                    "memory_id": {"type": "string", "description": "O ID da memória a ser usada."}
                },
                "required": ["query", "memory_id"]
            }
        }
    ]
    
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"tools": tools}
    }

async def handle_call_tool(params: Dict[str, Any], request_id):
    """Executa uma ferramenta específica"""
    tool_name = params.get("name")
    arguments = params.get("arguments", {})
    
    if tool_name == "create_memory_from_file":
        result = await tool_create_memory(arguments)
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"content": [{"type": "text", "text": json.dumps(result)}]}
        }
    elif tool_name == "chat_with_memory":
        # Chat com streaming direto na resposta MCP
        result = await tool_chat_with_memory_streaming(arguments)
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"content": [{"type": "text", "text": result}]}
        }
    else:
        return create_error_response(request_id, -32602, f"Ferramenta desconhecida: {tool_name}")

def create_error_response(request_id, code: int, message: str):
    """Cria uma resposta de erro no formato MCP padrão"""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message
        }
    }

# --- FERRAMENTAS ---
async def tool_create_memory(args: Dict[str, Any]) -> Dict[str, Any]:
    """Cria uma memória a partir de um arquivo"""
    filename = args.get("filename")
    file_content_base64 = args.get("file_content_base64")
    
    if not filename or not file_content_base64:
        raise HTTPException(400, "Parâmetros 'filename' e 'file_content_base64' são obrigatórios.")

    memory_id = str(uuid.uuid4())
    temp_file_path = OUTPUT_DIR / f"temp_{memory_id}_{Path(filename).name}"
    
    try:
        # Decodifica e salva o arquivo temporário
        file_content = base64.b64decode(file_content_base64)
        temp_file_path.write_bytes(file_content)

        # Cria o encoder
        encoder = MemvidEncoder()
        if temp_file_path.suffix.lower() == '.pdf':
            encoder.add_pdf(str(temp_file_path))
        else:
            encoder.add_text(temp_file_path.read_text(encoding='utf-8', errors='ignore'))
        
        if not encoder.chunks:
            raise HTTPException(400, "Nenhum conteúdo extraível encontrado no arquivo.")

        # Gera vídeo e índice
        video_path = OUTPUT_DIR / f"{memory_id}.{VIDEO_EXT}"
        index_path = OUTPUT_DIR / f"{memory_id}_index.json"
        encoder.build_video(str(video_path), str(index_path), codec=VIDEO_CODEC)

        logger.info(f"Memória criada com sucesso: {memory_id}")
        return {
            "memory_id": memory_id,
            "status": "success",
            "message": f"Memória criada com ID: {memory_id}"
        }
        
    except Exception as e:
        logger.error(f"Erro ao criar memória: {e}", exc_info=True)
        raise HTTPException(500, f"Erro ao criar memória: {e}")
    finally:
        # Limpa arquivo temporário
        if temp_file_path.exists():
            temp_file_path.unlink()

async def tool_chat_with_memory_streaming(args: Dict[str, Any]) -> str:
    """Executa chat com streaming e retorna resposta completa"""
    memory_id = args.get("memory_id")
    query = args.get("query")
    
    if not memory_id or not query:
        raise HTTPException(400, "Parâmetros 'memory_id' e 'query' são obrigatórios.")
    
    try:
        video_path = OUTPUT_DIR / f"{memory_id}.{VIDEO_EXT}"
        index_path = OUTPUT_DIR / f"{memory_id}_index.json"
        
        if not video_path.exists() or not index_path.exists():
            return json.dumps({"error": "Memory ID not found"})

        chat = MemvidChat(str(video_path), str(index_path), llm_provider=LLM_PROVIDER)
        
        # Coleta toda a resposta do streaming
        full_response = ""
        stream = chat.chat(query, stream=True)
        
        for chunk in stream:
            if chunk:
                full_response += chunk
        
        result = {
            "memory_id": memory_id,
            "query": query,
            "response": full_response,
            "status": "completed"
        }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Erro no chat: {e}", exc_info=True)
        return json.dumps({"error": str(e), "status": "error"})

# --- ROTAS SSE ---
@app.get("/chat/stream/{memory_id}")
async def stream_chat(memory_id: str, query: str):
    """Endpoint para streaming de chat usando SSE"""
    if not query:
        raise HTTPException(400, "Parâmetro 'query' é obrigatório.")
    
    return StreamingResponse(
        sse_chat_generator(memory_id, query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

async def sse_chat_generator(memory_id: str, query: str) -> Generator[str, None, None]:
    """Gerador para streaming SSE do chat"""
    try:
        video_path = OUTPUT_DIR / f"{memory_id}.{VIDEO_EXT}"
        index_path = OUTPUT_DIR / f"{memory_id}_index.json"
        
        if not video_path.exists() or not index_path.exists():
            yield f"data: {json.dumps({'error': 'Memory ID not found'})}\n\n"
            return

        chat = MemvidChat(str(video_path), str(index_path), llm_provider=LLM_PROVIDER)
        stream = chat.chat(query, stream=True)

        yield f"data: {json.dumps({'event': 'stream_start'})}\n\n"
        
        for chunk in stream:
            if chunk:
                yield f"data: {json.dumps({'token': chunk})}\n\n"
        
        yield f"data: {json.dumps({'event': 'stream_end'})}\n\n"
        
    except Exception as e:
        logger.error(f"Erro no streaming: {e}", exc_info=True)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        yield f"data: {json.dumps({'event': 'close'})}\n\n"

# --- ROTAS DE SAÚDE ---
@app.get("/health")
async def health_check():
    """Endpoint de verificação de saúde"""
    return {"status": "healthy", "service": "Memvid MCP Server"}

@app.get("/")
async def root():
    """Endpoint raiz com informações do servidor"""
    return {
        "service": "Memvid MCP Server",
        "version": "1.0.0",
        "endpoints": {
            "mcp": "/mcp",
            "stream": "/chat/stream/{memory_id}?query=...",
            "health": "/health"
        }
    }

# --- INICIALIZAÇÃO ---
if __name__ == "__main__":
    logger.info(f"🚀 Servidor MCP Memvid iniciando em {HOST}:{PORT}")
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
        access_log=True,
        log_level="info"
    )