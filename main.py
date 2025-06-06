# mcp_server.py
#
# SERVIDOR DE API AVANÇADO PARA MEMVID (Versão FastAPI Final - CORRIGIDA)
#
# DESCRIÇÃO:
# API robusta para criar "memórias de sessão" a partir de arquivos e conversar
# com elas. O endpoint /chat agora requer um `memory_id`, garantindo que toda
# conversa seja contextualizada por um documento processado.
#
# FLUXO DE TRABALHO:
# 1. POST /upload: Cliente envia um arquivo (PDF, TXT, etc.). Servidor processa
#    com Memvid e retorna um `memory_id`.
# 2. GET /chat?query=...&memory_id=...: Cliente envia a query e o ID da memória.
#    O servidor usa o MemvidChat para encontrar contexto e transmite a resposta do LLM.
#
# DEPENDÊNCIAS:
# pip install "memvid[llm,web]" uvicorn python-multipart
#
# VARIÁVEIS DE AMBIENTE (a serem configuradas no Render):
# GOOGLE_API_KEY, OPENAI_API_KEY, ou ANTHROPIC_API_KEY

import os
import asyncio
import json
import uuid
import logging
from pathlib import Path
from typing import AsyncGenerator, Optional, Generator
import traceback

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Importações seguras dos módulos Memvid
try:
    from memvid import MemvidEncoder
    from memvid import MemvidChat
    from memvid.config import get_codec_parameters
    from memvid.retriever import MemvidRetriever
    from memvid.llm_client import LLMClient
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logging.error(f"Erro crítico: Não foi possível importar os módulos do Memvid. {e}")
    logging.error("Certifique-se de que a biblioteca 'memvid' está instalada corretamente.")
    exit(1)

# --- CONFIGURAÇÃO DE LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURAÇÃO GLOBAL ---
OUTPUT_DIR = Path("session_memories")
VIDEO_CODEC = "mp4v"  # Codec leve que não requer FFmpeg/Docker
LLM_PROVIDER = os.getenv("MEMVID_PROVIDER", "google")

# Validação da chave de API na inicialização
try:
    from memvid.llm_client import LLMClient
    if not LLMClient.check_api_keys().get(LLM_PROVIDER, False):
        logger.warning(f"AVISO: Chave de API para o provedor '{LLM_PROVIDER}' não encontrada. O servidor pode não funcionar.")
        logger.warning("Defina a variável de ambiente apropriada (ex: GOOGLE_API_KEY).")
except Exception as e:
    logger.error(f"Erro ao verificar chaves de API: {e}")

try:
    VIDEO_EXT = get_codec_parameters(VIDEO_CODEC)["video_file_type"]
except KeyError:
    logger.error(f"Codec '{VIDEO_CODEC}' não encontrado na configuração. Usando '.mp4' como fallback.")
    VIDEO_EXT = "mp4"

app = FastAPI(
    title="Memvid API Server",
    description="API para processar documentos e interagir com eles usando a memória de vídeo do Memvid.",
    version="1.1.0"
)

OUTPUT_DIR.mkdir(exist_ok=True)

# --- MIDDLEWARE DE CORS ---
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://127.0.0.1:5500",
    "null",
    "*",  # Para desenvolvimento - remover em produção
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# --- ENDPOINTS DA API ---

@app.get("/")
async def get_root():
    """Endpoint raiz para verificação de status."""
    return {"status": "online", "message": "Servidor de API Memvid está rodando."}

@app.get("/health")
async def health_check():
    """Endpoint de verificação de saúde para serviços como o Render."""
    return {"status": "healthy"}

@app.post("/upload")
async def upload_and_create_memory(file: UploadFile = File(...)):
    """Recebe um arquivo, processa com Memvid e retorna um ID de memória."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nome do arquivo não fornecido.")

    # Gera um ID único e seguro para esta sessão de memória
    memory_id = str(uuid.uuid4())
    temp_file_path = OUTPUT_DIR / f"temp_{memory_id}_{Path(file.filename).name}"
    video_path = OUTPUT_DIR / f"{memory_id}.{VIDEO_EXT}"
    index_path = OUTPUT_DIR / f"{memory_id}_index.json"

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Arquivo enviado está vazio.")

        with open(temp_file_path, "wb") as buffer:
            buffer.write(content)

        logger.info(f"📄 Arquivo recebido: {file.filename} | Tamanho: {len(content) / 1024:.2f} KB")
        logger.info(f"🧠 Processando para memory_id: {memory_id}")

        encoder = MemvidEncoder()
        file_ext = Path(file.filename).suffix.lower()

        if file_ext == '.pdf':
            encoder.add_pdf(str(temp_file_path))
        elif file_ext in ['.txt', '.md']:
            encoder.add_text(temp_file_path.read_text(encoding='utf-8', errors='ignore'))
        else:
            raise HTTPException(status_code=415, detail=f"Tipo de arquivo não suportado: {file_ext}")
        
        if not encoder.chunks:
            raise HTTPException(status_code=400, detail="Nenhum conteúdo de texto pôde ser extraído do arquivo.")

        encoder.build_video(str(video_path), str(index_path), codec=VIDEO_CODEC)
        logger.info(f"✅ Memória '{memory_id}' criada com {len(encoder.chunks)} chunks.")

        return {"memory_id": memory_id, "filename": file.filename, "chunks_created": len(encoder.chunks)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro ao processar o arquivo para memory_id '{memory_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Falha ao processar o arquivo. {e}")
    finally:
        if temp_file_path.exists():
            temp_file_path.unlink()

def sse_chat_streamer(query: str, memory_id: str) -> Generator[str, None, None]:
    """
    Gerador SÍNCRONO que transmite a resposta do chat com tratamento robusto de erros.
    """
    try:
        # Verificar se os arquivos de memória existem
        video_path = OUTPUT_DIR / f"{memory_id}.{VIDEO_EXT}"
        index_path = OUTPUT_DIR / f"{memory_id}_index.json"

        if not video_path.exists() or not index_path.exists():
            error_msg = f"Arquivos de memória para o ID '{memory_id}' não encontrados."
            logger.error(error_msg)
            yield f"data: {json.dumps({'error': error_msg, 'code': 'MEMORY_NOT_FOUND'})}\n\n"
            return
        
        logger.info(f"🎬 Usando memória: {memory_id} com provedor: {LLM_PROVIDER}")
        
        # Verificar se a chave da API está configurada
        try:
            llm_client = LLMClient(provider=LLM_PROVIDER)
        except Exception as e:
            error_msg = f"Erro ao inicializar o cliente LLM: {str(e)}"
            logger.error(error_msg)
            yield f"data: {json.dumps({'error': error_msg, 'code': 'LLM_INIT_ERROR'})}\n\n"
            return
        
        # PASSO 1: Obter o contexto usando o Retriever
        try:
            retriever = MemvidRetriever(video_file=str(video_path), index_file=str(index_path))
            context_chunks = retriever.search(query)
            context_text = "\n".join(context_chunks)
            logger.info(f"🧠 Contexto encontrado: {len(context_text)} caracteres.")
        except Exception as e:
            error_msg = f"Erro ao buscar contexto: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield f"data: {json.dumps({'error': error_msg, 'code': 'RETRIEVER_ERROR'})}\n\n"
            return
        
        # PASSO 2: Preparar o prompt para o LLM
        full_prompt_content = (
            "Você é um assistente prestativo. Responda à pergunta do usuário com base apenas no contexto fornecido. "
            "Se a resposta não estiver no contexto, diga que você não sabe.\n\n"
            f"Contexto:\n---\n{context_text}\n---\n\n"
            f"Pergunta: {query}\n\nResposta:"
        )

        # Estrutura de mensagem correta para a API
        messages = [
            {'role': 'user', 'content': full_prompt_content}
        ]
        
        # PASSO 3: Iniciar o streaming
        yield f"data: {json.dumps({'event': 'stream_start', 'memory_id': memory_id})}\n\n"
        
        try:
            # Chamar o LLM com streaming
            stream_iterator = llm_client.chat(messages, stream=True)
            
            # Verificar se o stream_iterator não é None
            if stream_iterator is None:
                error_msg = "O cliente LLM retornou None. Verifique a configuração da API."
                logger.error(error_msg)
                yield f"data: {json.dumps({'error': error_msg, 'code': 'LLM_STREAM_NULL'})}\n\n"
                return
            
            # Iterar sobre as respostas do LLM
            token_count = 0
            for chunk in stream_iterator:
                if chunk:
                    token_count += 1
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
                    
            logger.info(f"✅ Stream concluído com {token_count} tokens para memory_id '{memory_id}'")
            
        except TypeError as e:
            # Captura o erro "'NoneType' object is not iterable"
            error_msg = f"Erro de iteração do LLM: {str(e)}. Verifique as configurações da API."
            logger.error(error_msg, exc_info=True)
            yield f"data: {json.dumps({'error': error_msg, 'code': 'LLM_ITERATION_ERROR'})}\n\n"
            return
            
    except Exception as e:
        error_msg = f"Erro geral durante o streaming: {str(e)}"
        logger.error(error_msg, exc_info=True)
        yield f"data: {json.dumps({'error': error_msg, 'code': 'STREAM_ERROR', 'traceback': traceback.format_exc()})}\n\n"
    finally:
        yield f"data: {json.dumps({'event': 'stream_end'})}\n\n"


@app.get("/chat")
async def chat_endpoint(request: Request, query: str, memory_id: str):
    """Endpoint de chat com validação robusta de parâmetros."""
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="O parâmetro 'query' é obrigatório e não pode ser vazio.")
    
    if not memory_id or not memory_id.strip():
        raise HTTPException(status_code=400, detail="O parâmetro 'memory_id' é obrigatório para o chat.")

    logger.info(f"💬 Query recebida: '{query[:100]}...' | memory_id: {memory_id}")

    # Verificar se a memória existe antes de iniciar o streaming
    video_path = OUTPUT_DIR / f"{memory_id}.{VIDEO_EXT}"
    index_path = OUTPUT_DIR / f"{memory_id}_index.json"
    
    if not video_path.exists() or not index_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Memória com ID '{memory_id}' não encontrada. Faça upload de um arquivo primeiro."
        )

    return StreamingResponse(
        sse_chat_streamer(query, memory_id), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """Endpoint para remover arquivos de uma memória específica."""
    logger.info(f"🗑️ Recebida requisição para deletar a memória: {memory_id}")
    video_path = OUTPUT_DIR / f"{memory_id}.{VIDEO_EXT}"
    index_path = OUTPUT_DIR / f"{memory_id}_index.json"
    faiss_path = OUTPUT_DIR / f"{memory_id}_index.faiss"
    
    deleted_files_count = 0
    for file_path in [video_path, index_path, faiss_path]:
        if file_path.exists():
            try:
                file_path.unlink()
                deleted_files_count += 1
                logger.info(f"   - Arquivo removido: {file_path.name}")
            except OSError as e:
                logger.error(f"Erro ao remover o arquivo {file_path}: {e}")
                # Continua tentando remover os outros arquivos
    
    if deleted_files_count == 0:
        raise HTTPException(status_code=404, detail="Nenhum arquivo encontrado para esta memória.")
    
    return {"message": f"Memória {memory_id} e seus arquivos associados foram removidos."}

# Endpoint para listar memórias (útil para debug)
@app.get("/memories")
async def list_memories():
    """Lista todas as memórias disponíveis."""
    memories = []
    for video_file in OUTPUT_DIR.glob(f"*.{VIDEO_EXT}"):
        memory_id = video_file.stem
        index_file = OUTPUT_DIR / f"{memory_id}_index.json"
        if index_file.exists():
            memories.append({
                "memory_id": memory_id,
                "video_file": video_file.name,
                "index_file": index_file.name,
                "created": video_file.stat().st_ctime
            })
    
    return {"memories": memories, "count": len(memories)}

# --- BLOCO DE EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Iniciando servidor Memvid em http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)