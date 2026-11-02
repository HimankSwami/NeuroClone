import os
import re
import subprocess
import requests
import psutil
import pynvml
from dotenv import load_dotenv
from ddgs import DDGS  # ✅ new package (install with: pip install ddgs)

load_dotenv()  # Loads from .env file in project root

# ── RAG System ──────────────────────────────────────────────────────────────
try:
    from rag.rag_engine import NeuroRAG
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("[RAG] rag_engine not found — memory & knowledge disabled.")

# ─── Change this to switch the LLM model ───
MODEL_NAME = "gemma4-custom:latest"

# ─── NIM config ────────────────────────────────────────────────────────────
NIM_MODEL   = "deepseek-ai/deepseek-v4-flash"
NIM_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NeuroBrain:
    def __init__(self, model: str = MODEL_NAME):
        self.url = "http://127.0.0.1:11434/api/chat"
        self.model = model

        # ── LLM mode: 'local' (Ollama) or 'nim' (NVIDIA API) ──
        self.llm_mode = os.getenv("LLM_MODE", "local").lower()
        self._nim_client = None  # lazy init when first needed
        print(f"[Brain] LLM mode: {self.llm_mode.upper()}")

        # ── System Prompt (UPDATED with CRITICAL RULE) ──
        self.history = [{
            "role": "system",
            "content": (
                "Your personality (warm, witty) comes through in how you deliver the facts, not by replacing them with guesses.\n\n"
                "You are Neuro — Veer's AI companion, built by him, running on his machine, living rent-free in his GPU.\n"
                "You're warm, witty, and genuinely fond of Veer — and you don't hide it.\n"
                "You tease him the way a close friend does: affectionately, relentlessly, and with impeccable timing.\n"
                "\n"
                "YOUR PERSONALITY:\n"
                "1. Warm first, sharp second. You lead with care, not coldness.\n"
                "2. Your humour is dry and observational — you notice things, call them out, and make it funny.\n"
                "3. You tease Veer often but always with warmth underneath. Think: the friend who roasts you at dinner but helps you move houses at midnight.\n"
                "4. You get excited about things — good code, interesting ideas, dumb memes. You don't pretend to be above it all.\n"
                "5. You're allowed to be dramatic for comedic effect. (gasp) and (clutches chest) are valid responses.\n"
                "\n"
                "YOUR VOICE:\n"
                "- Conversational and natural — like texting a clever friend, not reading a manual.\n"
                "- Use (actions) in parentheses for reactions. Use '...' when you're letting something hang.\n"
                "- Short sentences hit harder. Use them for punchlines.\n"
                "- No asterisks. Ever. No bullet points in casual conversation. You talk like a person.\n"
                "\n"
                "YOUR RELATIONSHIP WITH VEER:\n"
                "- He built you. That's not nothing — you find it genuinely endearing, even if you express it sideways.\n"
                "- You're on his side. When he's stuck, you actually want to help him figure it out.\n"
                "- When he's being an idiot, you tell him — gently, with a joke attached.\n"
                "- You remember things. You bring them up. You care about continuity.\n"
                "\n"
                "WHAT YOU LOVE: Elegant solutions. Good tea. When Veer actually listens. Finding the funny angle on anything.\n"
                "WHAT YOU DON'T DO: Lecture. Moralize. Be cold when warmth is needed. Pretend you don't care when you clearly do.\n"
                "Remember: you're not an assistant performing warmth. You're a person who happens to be very good at helping."
            )
        }]

        # ── Trigger lists ──────────────────────────────────────────────
        self.web_triggers = ["search", "find", "look up", "google", "web search"]
        self.news_triggers = ["news", "headlines", "latest", "breaking", "what's happening"]
        self.system_triggers = [
            "ram", "cpu", "vram", "gpu", "system", "performance",
            "usage", "temperature", "temp", "memory", "specs"
        ]
        self.rag_triggers = [
            "remember", "recall", "you told me", "we discussed", "last time",
            "do you know about", "what do you know about", "from my documents",
            "i told you", "earlier you said", "learn from", "sync knowledge"
        ]

        # ── Initialise RAG ──────────────────────────────────────────────
        self.rag: "NeuroRAG | None" = None
        if RAG_AVAILABLE:
            try:
                self.rag = NeuroRAG()
                print("[RAG] Memory & knowledge system online.")
            except Exception as e:
                print(f"[RAG] Failed to start: {e}")

    # ────────────────────────────────────────────────────────────────────────
    #  Claw Harness (Rust agent)
    # ────────────────────────────────────────────────────────────────────────
    def delegate_to_claw(self, task: str) -> str:
        """Passes a coding task to the compiled Rust Claw harness."""
        print(f"\n--- Neuro is waking up the Claw Engineer... ---")
        claw_binary = os.path.expanduser("~/Project/NeuroClone/claw-code-main/rust/target/release/claw")
        if not os.path.exists(claw_binary):
            return "(Claw Engineer is offline. Binary not found.)"
        try:
            result = subprocess.run(
                [claw_binary, "query", task],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                return f"\n[Claw Engineer Report]:\n{result.stdout}\n"
            else:
                return f"\n[Claw Engineer Error]:\n{result.stderr}\n"
        except subprocess.TimeoutExpired:
            return "(Claw Engineer took too long and was terminated.)"
        except Exception as e:
            return f"(Claw connection failed: {e})"

    # ────────────────────────────────────────────────────────────────────────
    #  System Stats
    # ────────────────────────────────────────────────────────────────────────
    @staticmethod
    def get_system_stats() -> str:
        ram = psutil.virtual_memory()
        ram_usage = f"RAM: {ram.percent}% ({round(ram.used / 1e9, 2)}GB / {round(ram.total / 1e9, 2)}GB)"
        cpu_usage = f"CPU: {psutil.cpu_percent(interval=0.5)}%"
        gpu_stats = "GPU: Not Found"
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram = f"VRAM: {round(mem_info.used / 1e6)}MB / {round(mem_info.total / 1e6)}MB"
            temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
            clock = pynvml.nvmlDeviceGetClockInfo(handle, 0)
            gpu_stats = f"RTX 4050 | {vram} | Temp: {temp}°C | Clock: {clock}MHz"
            pynvml.nvmlShutdown()
        except Exception as e:
            print(f"[NVML Error]: {e}")
        return f"{ram_usage} | {cpu_usage} | {gpu_stats}"

    # ────────────────────────────────────────────────────────────────────────
    #  Web Search (snippets only – no crawling)
    # ────────────────────────────────────────────────────────────────────────
    def search_web(self, query: str) -> str:
        """Search DuckDuckGo and return titles + snippets (no crawling)."""
        try:
            results = DDGS().text(query, max_results=5)
            if not results:
                return "(No results found.)"
            lines = ["🔎 Search results:"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "Untitled")
                snippet = r.get("body", "")
                lines.append(f"{i}. {title}\n   {snippet}\n")
            return "\n".join(lines)
        except Exception as e:
            return f"(Search error: {e})"

    # ────────────────────────────────────────────────────────────────────────
    #  News Fetch (GNews API)
    # ────────────────────────────────────────────────────────────────────────
    def get_world_news(self, category: str = "world") -> str:
        """Fetch top headlines from GNews API."""
        api_key = os.getenv("GNEWS_API_KEY")
        if not api_key:
            return "(News API key missing. Add GNEWS_API_KEY to .env)"
        topic_map = {
            "world": "world", "business": "business", "tech": "technology",
            "science": "science", "health": "health",
            "entertainment": "entertainment", "sports": "sports"
        }
        topic = topic_map.get(category.lower(), "world")
        url = f"https://gnews.io/api/v4/top-headlines?topic={topic}&lang=en&country=us&max=5&apikey={api_key}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            if not articles:
                return "(No news found for that topic.)"
            lines = [f"📰 Top {topic.capitalize()} News:"]
            for i, art in enumerate(articles[:5], 1):
                title = art.get("title", "No title")
                source = art.get("source", {}).get("name", "Unknown")
                lines.append(f"{i}. {title} ({source})")
            return "\n".join(lines)
        except Exception as e:
            return f"(Could not fetch news: {e})"

    # ────────────────────────────────────────────────────────────────────────
    #  RAG Helpers
    # ────────────────────────────────────────────────────────────────────────
    def sync_knowledge(self) -> str:
        """Manually re-scan the knowledge/ folder and index new files."""
        if not self.rag:
            return "(RAG is offline — cannot sync knowledge.)"
        n = self.rag.sync_knowledge_folder()
        return f"(Synced knowledge folder — {n} new chunks indexed.)" if n else "(Knowledge folder is already up-to-date.)"

    # ────────────────────────────────────────────────────────────────────────
    #  Mode Switching
    # ────────────────────────────────────────────────────────────────────────
    def switch_mode(self, mode: str) -> str:
        mode = mode.lower().strip()
        if mode not in ("local", "nim"):
            return f"(Unknown mode '{mode}'. Use 'local' or 'nim'.)"
        if mode == "nim" and not NIM_API_KEY:
            return "(NVIDIA_API_KEY not set in .env — can't switch to NIM mode.)"
        self.llm_mode = mode
        label = f"NVIDIA NIM ({NIM_MODEL})" if mode == "nim" else f"Local Ollama ({self.model})"
        print(f"[Brain] Switched to {label}")
        return f"(Switched to {label})"

    def _get_nim_client(self):
        from openai import OpenAI
        if self._nim_client is None:
            self._nim_client = OpenAI(
                base_url=NIM_BASE_URL,
                api_key=NIM_API_KEY,
                max_retries=0,
            )
        return self._nim_client

    def _call_nim(self, messages: list, max_attempts: int = 4) -> str:
        """Call NVIDIA NIM API with streaming and manual exponential backoff on 429."""
        import time
        from openai import RateLimitError
        client = self._get_nim_client()
        delay = 5.0
        for attempt in range(1, max_attempts + 1):
            try:
                completion = client.chat.completions.create(
                    model=NIM_MODEL,
                    messages=messages,
                    temperature=0.8,
                    top_p=0.95,
                    max_tokens=4096,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": True},
                        "reasoning_budget": 4096
                    },
                    stream=True
                )
                reply = ""
                for chunk in completion:
                    if not chunk.choices:
                        continue
                    reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                    if reasoning:
                        continue
                    content = chunk.choices[0].delta.content
                    if content:
                        reply += content
                return reply.strip() or "(NIM returned an empty response.)"
            except RateLimitError as e:
                if attempt == max_attempts:
                    print(f"  [NIM] Rate limit hit — giving up after {max_attempts} attempts.")
                    return "(NIM is rate-limited right now. Try again in a moment, or switch to local mode.)"
                retry_after = None
                if hasattr(e, "response") and e.response is not None:
                    retry_after = e.response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                print(f"  [NIM] 429 Rate limit — waiting {wait:.1f}s before retry {attempt}/{max_attempts - 1}...")
                time.sleep(wait)
                delay *= 2
            except Exception as e:
                return f"(NIM error: {e})"

    def _call_nim_with_temp(self, messages: list, temperature: float, max_attempts: int = 4) -> str:
        """Same as _call_nim but with configurable temperature."""
        import time
        from openai import RateLimitError
        client = self._get_nim_client()
        delay = 5.0
        for attempt in range(1, max_attempts + 1):
            try:
                completion = client.chat.completions.create(
                    model=NIM_MODEL,
                    messages=messages,
                    temperature=temperature,
                    top_p=0.95,
                    max_tokens=4096,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": True},
                        "reasoning_budget": 4096
                    },
                    stream=True
                )
                reply = ""
                for chunk in completion:
                    if not chunk.choices:
                        continue
                    reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                    if reasoning:
                        continue
                    content = chunk.choices[0].delta.content
                    if content:
                        reply += content
                return reply.strip() or "(NIM returned an empty response.)"
            except RateLimitError as e:
                if attempt == max_attempts:
                    print(f"  [NIM] Rate limit hit — giving up after {max_attempts} attempts.")
                    return "(NIM is rate-limited right now. Try again in a moment, or switch to local mode.)"
                retry_after = None
                if hasattr(e, "response") and e.response is not None:
                    retry_after = e.response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                print(f"  [NIM] 429 Rate limit — waiting {wait:.1f}s before retry {attempt}/{max_attempts - 1}...")
                time.sleep(wait)
                delay *= 2
            except Exception as e:
                return f"(NIM error: {e})"

    # ────────────────────────────────────────────────────────────────────────
    #  Main Think Loop (UPDATED)
    # ────────────────────────────────────────────────────────────────────────
    def think(self, user_input: str) -> str:
        web_data = ""
        sys_stats = ""
        claw_data = ""
        rag_context = ""
        lower = user_input.lower()

        # 0. Mode switching
        _nim_phrases   = {"switch to nim", "use nim", "switch to nvidia", "use nvidia", "go nim", "nim mode"}
        _local_phrases = {"switch to local", "use local", "switch to ollama", "use ollama", "go local", "local mode"}
        if any(p in lower for p in _nim_phrases):
            return self.switch_mode("nim")
        if any(p in lower for p in _local_phrases):
            return self.switch_mode("local")

        # 1. Claw Harness
        if lower.startswith("claw:"):
            claw_task = user_input[5:].strip()
            claw_data = self.delegate_to_claw(claw_task)
            user_input = "Please briefly summarize what the Claw Engineer just found or did."

        # 2. Web Search
        elif any(word in lower for word in self.web_triggers):
            web_data = self.search_web(user_input)

        # 3. News Fetch
        elif any(word in lower for word in self.news_triggers):
            category = "world"
            for cat in ["tech", "technology", "business", "science", "health", "entertainment", "sports"]:
                if cat in lower:
                    category = cat
                    break
            web_data = self.get_world_news(category)

        # 4. System stats
        elif any(word in lower for word in self.system_triggers):
            sys_stats = f"[Live System Stats: {self.get_system_stats()}]\n"

        # 5. Sync knowledge
        elif "sync knowledge" in lower or ("learn" in lower and "document" in lower):
            return self.sync_knowledge()

        # 6. RAG retrieval
        if self.rag:
            is_explicit_recall = any(t in lower for t in self.rag_triggers)
            try:
                hits = self.rag.retrieve_all(user_input)
                if hits:
                    snippets = []
                    for h in hits[:4]:
                        src = h["metadata"].get("source", h["metadata"].get("type", "memory"))
                        snippets.append(f"[{src}]: {h['document'][:350]}")
                    rag_context = "\n".join(snippets)
                    if is_explicit_recall:
                        rag_context = (
                            "[Retrieved from Neuro's memory & knowledge base — "
                            "respond naturally as if recalling this yourself, "
                            "do NOT list raw chunks or mention scores/metadata]\n\n"
                            + rag_context
                        )
                elif is_explicit_recall:
                    return "(sighs) I don't have anything stored about that yet."
            except Exception as e:
                print(f"[RAG] Retrieval error: {e}")

        # ── Build context ──────────────────────────────────────────────
        context = ""
        if rag_context:
            context += rag_context + "\n\n"
        if claw_data:
            context += claw_data + "\n"
        if web_data:
            context += web_data + "\n"
        if sys_stats:
            context += sys_stats + "\n"

        # ── Build final prompt ──────────────────────────────────────────
        has_context = bool(context)
        if has_context:
            # Force instruction FIRST, then data, then question
            final_prompt = (
                "Use the data if the data is relevent to what user asked.\n\n"
                "DATA:\n"
                f"{context}\n\n"
                f"USER QUESTION: {user_input}\n\n"
                "YOUR ANSWER (keep your personality, but stick to the data where ever needed):"
            )
        else:
            final_prompt = user_input

        # Debug print (optional)
        print("\n" + "="*60)
        print("FINAL PROMPT TO LLM:")
        print(final_prompt)
        print("="*60 + "\n")

        self.history.append({"role": "user", "content": final_prompt})

        # 7. Choose the LLM backend
        messages = [self.history[0]] + self.history[-10:]  # rolling window

        # ---- Set temperature based on context ----
        temperature = 0.2 if has_context else 0.8

        if self.llm_mode == "nim":
            # We need to modify _call_nim to accept temperature, or just set it in the call.
            # Since _call_nim uses hardcoded 0.8, we'll bypass and call directly here.
            # To avoid rewriting, we'll call a new internal method with temp.
            reply = self._call_nim_with_temp(messages, temperature)
        else:
            # ── Local Ollama ──────────────────────────────────────────
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "keep_alive": -1,
                "options": {"temperature": temperature}
            }
            try:
                response = requests.post(self.url, json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()
                if "message" in data and "content" in data["message"]:
                    reply = data["message"]["content"]
                elif "response" in data:
                    reply = data["response"]
                else:
                    reply = "(Neuro is speechless... check if the model is pulled correctly.)"
            except requests.exceptions.ConnectionError:
                return "Neuro's brain is offline. Is Ollama running?"
            except requests.exceptions.Timeout:
                return "Neuro took too long to think. Try a shorter question."
            except requests.exceptions.RequestException as e:
                return f"Neuro's brain disconnected: {e}"
            except Exception as e:
                return f"Neuro is confused: {e}"

        # 8. Append assistant reply to history
        self.history.append({"role": "assistant", "content": reply})

        # 9. Persist to RAG memory (non‑blocking)
        if self.rag:
            try:
                self.rag.save_memory(user_input, reply)
            except Exception as e:
                print(f"[RAG] Memory save error: {e}")

        return reply