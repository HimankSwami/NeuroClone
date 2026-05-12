import asyncio
import os
import re
import sys
import threading
import uvicorn
# Add the project root to sys.path so brain/ is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["PYTHONPATH"] = PROJECT_ROOT

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from brain.core import NeuroBrain, MODEL_NAME
from voice.speaker import speak

app = FastAPI(title="Neuro VTuber Interface")
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")

brain = NeuroBrain(model=MODEL_NAME)
voice_enabled = True  # toggle with "voice" / "text" commands


def _speak_thread(text: str):
    """Run blocking speak() in a background thread."""
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
    return {"status": "online", "model": brain.model}


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

    try:
        while True:
            data = await websocket.receive_json()
            user_input = data.get("message", "").strip()

            if not user_input:
                continue

            # Voice toggle commands
            lower = user_input.lower()
            if lower == "voice":
                voice_enabled = True
                await websocket.send_json({"type": "announcement", "text": "Voice responses enabled"})
                continue
            if lower == "text":
                voice_enabled = False
                await websocket.send_json({"type": "announcement", "text": "Text-only mode enabled"})
                continue

            await websocket.send_json({"type": "state", "state": "thinking"})

            response = brain.think(user_input)
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

            # Stream text word-by-word for avatar animation
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

            # Speak the response through Piper TTS → RVC (non-blocking)
            if voice_enabled:
                print(f"  [TTS] Speaking response...")
                threading.Thread(target=_speak_thread, args=(response,), daemon=True).start()

            await websocket.send_json({"type": "state", "state": "idle"})

    except WebSocketDisconnect:
        print("Client disconnected from WebSocket.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
