import os
import asyncio
import json
import uuid
import logging
from pathlib import Path
from typing import AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Importações do seu projeto memvid
try:
    from memvid.encoder import MemvidEncoder
    from memvid.chat import MemvidChat
    from memvid.config import get_codec_parameters
    from memvid.llm_client import LLMClient, create_llm_client
except ImportError as e:
    logging.error(f"Erro ao importar módulos memvid: {e}")
    raise

# --- CONFIGURAÇÃO DE LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURAÇÃO GLOBAL ---
OUTPUT_DIR = Path("session_memories")
VIDEO_CODEC = "mp4v"
LLM_PROVIDER = os.getenv("MEMVID_PROVIDER", "google")

# Verificação de variáveis de ambiente necessárias
required_env_vars = {
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY", 
    "anthropic": "ANTHROPIC_API_KEY"
}

if LLM_PROVIDER in required_env_vars:
    if not os.getenv(required_env_vars[LLM_PROVIDER]):
        logger.warning(f"Variável de ambiente {required_env_vars[LLM_PROVIDER]} não encontrada")

try:
    VIDEO_EXT = get_codec_parameters(VIDEO_CODEC)["video_file_type"]
except Exception as e:
    logger.error(f"Erro ao obter parâmetros do codec: {e}")
    VIDEO_EXT = "mp4"  # fallback

app = FastAPI(
    title="Memvid MCP Server",
    description="API para processamento de documentos e chat com contexto usando Memvid",
    version="1.0.0"
)

# Garante que o diretório de memórias exista
OUTPUT_DIR.mkdir(exist_ok=True)

# --- CONFIGURAÇÃO DE CORS ---
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "null",
    # Adicione sua URL de produção aqui quando necessário
    # "https://seu-cliente.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MIDDLEWARE DE LOGGING ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = asyncio.get_event_loop().time()
    response = await call_next(request)
    process_time = asyncio.get_event_loop().time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    return response

# --- ENDPOINTS DA API ---

@app.get("/")
async def get_root():
    """Endpoint raiz para verificação de status."""
    return {
        "status": "online", 
        "message": "Servidor de API Memvid está rodando.",
        "version": "1.0.0",
        "provider": LLM_PROVIDER
    }

@app.get("/health")
async def health_check():
    """Health check endpoint para o Render."""
    return {"status": "healthy", "timestamp": asyncio.get_event_loop().time()}

@app.post("/upload")
async def upload_and_create_memory(file: UploadFile = File(...)):
    """
    Recebe um arquivo, processa com Memvid e retorna um ID de memória.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nome do arquivo não fornecido.")

    # Verificação de tamanho do arquivo (limite de 50MB)
    if file.size and file.size > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Arquivo muito grande. Limite: 50MB")

    # Gera um ID único para esta sessão de memória
    memory_id = str(uuid.uuid4())
    temp_file_path = OUTPUT_DIR / f"temp_{memory_id}_{file.filename}"
    video_path = OUTPUT_DIR / f"{memory_id}.{VIDEO_EXT}"
    index_path = OUTPUT_DIR / f"{memory_id}_index.json"

    try:
        # Salva o arquivo temporariamente
        content = await file.read()
        with open(temp_file_path, "wb") as buffer:
            buffer.write(content)

        logger.info(f"📄 Arquivo recebido: {file.filename} ({file.content_type}) - {len(content)} bytes")
        logger.info(f"🧠 Processando para memory_id: {memory_id}")

        # Processa o arquivo com MemvidEncoder
        encoder = MemvidEncoder()
        file_ext = Path(file.filename).suffix.lower()

        if file_ext == '.pdf':
            encoder.add_pdf(str(temp_file_path))
        elif file_ext in ['.txt', '.md']:
            with open(temp_file_path, "r", encoding='utf-8') as f:
                content_text = f.read()
                if not content_text.strip():
                    raise HTTPException(status_code=400, detail="Arquivo de texto está vazio.")
                encoder.add_text(content_text)
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Tipo de arquivo não suportado: {file_ext}. Suportados: .pdf, .txt, .md"
            )
        
        if not encoder.chunks:
            raise HTTPException(status_code=400, detail="Nenhum conteúdo extraído do arquivo.")

        # Constrói o vídeo e o índice da memória
        encoder.build_video(str(video_path), str(index_path), codec=VIDEO_CODEC)
        logger.info(f"✅ Memória criada com sucesso: {video_path}")

        return {
            "memory_id": memory_id, 
            "filename": file.filename, 
            "chunks_created": len(encoder.chunks),
            "video_path": str(video_path.name),
            "index_path": str(index_path.name)
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"❌ Erro ao processar o arquivo: {e}")
        raise HTTPException(status_code=500, detail=f"Falha ao processar o arquivo: {str(e)}")
    finally:
        # Limpa o arquivo temporário
        if temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception as e:
                logger.warning(f"Erro ao remover arquivo temporário: {e}")
# No seu main.py, a função sse_chat_streamer
async def sse_chat_streamer(query: str, memory_id: Optional[str]) -> AsyncGenerator[str, None]:
    """
    Gerador que transmite a resposta do chat, usando Memvid se um memory_id for fornecido.
    """
    # Verificação robusta: memory_id é agora obrigatório para o chat
    if not memory_id:
        error_msg = "O 'memory_id' é obrigatório para o chat. Por favor, faça o upload de um documento primeiro."
        logger.error(error_msg)
        yield f"data: {json.dumps({'error': error_msg, 'code': 'MISSING_MEMORY_ID'})}\n\n"
        yield f"data: {json.dumps({'event': 'close'})}\n\n"
        return

    try:
        llm_provider = os.getenv("MEMVID_PROVIDER", "google")
        
        video_path = OUTPUT_DIR / f"{memory_id}.{VIDEO_EXT}"
        index_path = OUTPUT_DIR / f"{memory_id}_index.json"

        if not video_path.exists() or not index_path.exists():
            raise FileNotFoundError("ID de memória inválido ou arquivos não encontrados.")
        
        logger.info(f"🎬 Usando memória: {memory_id} com provedor: {llm_provider}")
        
        # Usando a sintaxe correta do file_chat.py
        chat_handler = MemvidChat(
            video_file=str(video_path),
            index_file=str(index_path),
            llm_provider=llm_provider
        )
        
        stream_iterator = chat_handler.chat(query, stream=True)

        # O resto do código permanece o mesmo
        yield f"data: {json.dumps({'event': 'start', 'memory_id': memory_id})}\n\n"
        
        for chunk in stream_iterator:
            if chunk:
                yield f"data: {json.dumps({'token': chunk, 'type': 'content'})}\n\n"
                await asyncio.sleep(0.001)

        yield f"data: {json.dumps({'event': 'end'})}\n\n"

    except FileNotFoundError as e:
        logger.error(f"Arquivo não encontrado: {e}")
        yield f"data: {json.dumps({'error': str(e), 'code': 'FILE_NOT_FOUND'})}\n\n"
    except Exception as e:
        logger.error(f"❌ Erro durante o streaming: {e}")
        yield f"data: {json.dumps({'error': str(e), 'code': 'STREAMING_ERROR'})}\n\n"
    finally:
        yield f"data: {json.dumps({'event': 'close'})}\n\n"

@app.get("/chat")
async def chat_endpoint(request: Request):
    """Endpoint SSE que recebe uma query e um memory_id opcional."""
    query = request.query_params.get('query')
    memory_id = request.query_params.get('memory_id')

    if not query:
        return JSONResponse(
            status_code=400, 
            content={"error": "O parâmetro 'query' é obrigatório."}
        )
    
    if not query.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Query não pode estar vazia."}
        )
    
    logger.info(f"💬 Query recebida: '{query[:100]}...' | memory_id: {memory_id or 'Nenhum'}")

    return StreamingResponse(
        sse_chat_streamer(query, memory_id), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.get("/memories")
async def list_memories():
    """Lista todas as memórias disponíveis."""
    memories = []
    for video_file in OUTPUT_DIR.glob(f"*.{VIDEO_EXT}"):
        memory_id = video_file.stem
        index_file = OUTPUT_DIR / f"{memory_id}_index.json"
        
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    index_data = json.load(f)
                    memories.append({
                        "memory_id": memory_id,
                        "created": video_file.stat().st_ctime,
                        "chunks": len(index_data.get('chunks', [])),
                        "video_size": video_file.stat().st_size
                    })
            except Exception as e:
                logger.warning(f"Erro ao ler índice {index_file}: {e}")
    
    return {"memories": memories, "total": len(memories)}

@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """Remove uma memória específica."""
    video_path = OUTPUT_DIR / f"{memory_id}.{VIDEO_EXT}"
    index_path = OUTPUT_DIR / f"{memory_id}_index.json"
    
    deleted_files = []
    
    if video_path.exists():
        video_path.unlink()
        deleted_files.append(str(video_path.name))
    
    if index_path.exists():
        index_path.unlink()
        deleted_files.append(str(index_path.name))
    
    if not deleted_files:
        raise HTTPException(status_code=404, detail="Memória não encontrada.")
    
    return {"message": f"Memória {memory_id} removida.", "deleted_files": deleted_files}

# --- TRATAMENTO DE ERROS GLOBAIS ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Erro não tratado: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Erro interno do servidor", "detail": str(exc)}
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )