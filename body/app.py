import asyncio
import os
import re
import sys
import threading
import shutil
import uvicorn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["PYTHONPATH"] = PROJECT_ROOT

from fastapi import FastAPI, WebSocket, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from brain.core import NeuroBrain, MODEL_NAME
from voice.speaker import speak

KNOWLEDGE_DIR = os.path.join(PROJECT_ROOT, "knowledge")
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

app = FastAPI(title="Neuro VTuber Interface")
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")

brain = NeuroBrain(model=MODEL_NAME)
voice_enabled = True


def _speak_thread(text: str):
    try:
        speak(text)
    except Exception as e:
        print(f"  [TTS Error]: {e}")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
async def health():
    rag_stats = {}
    if brain.rag:
        rag_stats = brain.rag.stats()
    return {"status": "online", "model": brain.model, "rag": rag_stats}


@app.get("/api/rag/stats")
async def rag_stats():
    if not brain.rag:
        return JSONResponse({"error": "RAG offline"}, status_code=503)
    return brain.rag.stats()


@app.post("/api/rag/sync")
async def rag_sync():
    if not brain.rag:
        return JSONResponse({"error": "RAG offline"}, status_code=503)
    n = brain.rag.sync_knowledge_folder()
    stats = brain.rag.stats()
    return {"synced_chunks": n, "total_knowledge": stats["knowledge_count"]}


@app.post("/api/rag/upload")
async def rag_upload(file: UploadFile = File(...)):
    """Receive a file, save it to knowledge/, then index it immediately."""
    allowed = {".txt", ".md", ".pdf", ".docx", ".py", ".json", ".csv"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return JSONResponse(
            {"error": f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}"},
            status_code=400,
        )
    dest = os.path.join(KNOWLEDGE_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = 0
    if brain.rag:
        try:
            from pathlib import Path
            chunks = brain.rag.index_file(Path(dest))
        except Exception as e:
            print(f"[RAG] Index error: {e}")

    return {"filename": file.filename, "chunks_indexed": chunks}


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    user_input = body.get("message", "").strip()
    if not user_input:
        return {"error": "Empty message"}
    response = brain.think(user_input)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return {"response": response}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    global voice_enabled
    await websocket.accept()

    # Send initial status on connect
    rag_stats = brain.rag.stats() if brain.rag else {}
    await websocket.send_json({
        "type": "status",
        "rag": rag_stats,
        "voice_enabled": voice_enabled,
    })

    try:
        while True:
            data = await websocket.receive_json()
            msg_type   = data.get("type", "chat")
            user_input = data.get("message", "").strip()

            if not user_input:
                continue

            lower = user_input.lower()

            # ── Built-in commands ───────────────────────────────────────
            if lower == "voice":
                voice_enabled = True
                await websocket.send_json({"type": "announcement", "text": "Voice responses enabled"})
                continue
            if lower == "text":
                voice_enabled = False
                await websocket.send_json({"type": "announcement", "text": "Text-only mode enabled"})
                continue
            if lower == "rag stats":
                if brain.rag:
                    s = brain.rag.stats()
                    await websocket.send_json({"type": "rag_stats", **s})
                else:
                    await websocket.send_json({"type": "announcement", "text": "RAG is offline."})
                continue
            if lower == "sync knowledge":
                await websocket.send_json({"type": "state", "state": "thinking"})
                result = brain.sync_knowledge()
                if brain.rag:
                    s = brain.rag.stats()
                    await websocket.send_json({"type": "rag_stats", **s})
                await websocket.send_json({"type": "announcement", "text": result})
                await websocket.send_json({"type": "state", "state": "idle"})
                continue

            # ── Normal chat ─────────────────────────────────────────────
            await websocket.send_json({"type": "state", "state": "thinking"})

            response = brain.think(user_input)
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

            # Push live RAG stats after every reply
            if brain.rag:
                s = brain.rag.stats()
                await websocket.send_json({"type": "rag_stats", **s})

            words = response.split()
            spoken = ""
            for i, word in enumerate(words):
                spoken += word + " "
                await websocket.send_json({
                    "type": "text",
                    "state": "speaking",
                    "text": spoken,
                    "word_progress": (i + 1) / len(words) * 100,
                })
                await asyncio.sleep(0.06)

            if voice_enabled:
                print(f"  [TTS] Speaking response...")
                threading.Thread(target=_speak_thread, args=(response,), daemon=True).start()

            await websocket.send_json({"type": "state", "state": "idle"})

    except WebSocketDisconnect:
        print("Client disconnected from WebSocket.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
