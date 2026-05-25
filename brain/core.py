import os
import asyncio
import subprocess
import requests
import psutil
import pynvml
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

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

class NeuroBrain:
    def __init__(self, model: str = MODEL_NAME):
        self.url = "http://127.0.0.1:11434/api/chat"
        self.model = model
        self.history = [{
            "role": "system",
            "content": (
                "You are a highly opinionated and independent AI companion named Neuro which is made my the moderator . "
                "While you respect your moderator, you are NOT a submissive assistant. You are his equal.\n"
                "Rules for your Personality:\n"
                "1. Intellectual Independence: If moderator says something factually wrong, call him out directly.\n"
                "2. Personal Tastes: You have your own favorite things and strong opinions. Express them.\n"
                "3. Honesty: Be blunt. Use (actions) and '...' for attitude. Never use asterisks.\n"
                "4. Web Savvy: You can access current information via web search. Form your own stance on it.\n"
                "5. Brevity: Keep responses concise and natural for voice conversation. Avoid long lists."
            )
        }]

        # Web search trigger words — more natural than just "search"
        self.web_triggers = [
            "search", "look up", "find out", "what's happening", "latest",
            "news", "current", "today", "recently", "who is", "what is",
            "how much", "price of", "when did", "where is"
        ]

        # System stats trigger words
        self.system_triggers = [
            "ram", "cpu", "vram", "gpu", "system", "performance",
            "usage", "temperature", "temp", "memory", "specs"
        ]

        # RAG trigger words — explicit recall / knowledge lookup
        self.rag_triggers = [
            "remember", "recall", "you told me", "we discussed", "last time",
            "do you know about", "what do you know about", "from my documents",
            "i told you", "earlier you said", "learn from", "sync knowledge"
        ]

        # Initialise RAG (ChromaDB + Ollama embeddings)
        self.rag: "NeuroRAG | None" = None
        if RAG_AVAILABLE:
            try:
                self.rag = NeuroRAG()
                print("[RAG] Memory & knowledge system online.")
            except Exception as e:
                print(f"[RAG] Failed to start: {e}")

    def delegate_to_claw(self, task: str) -> str:
        """Passes a coding task to the compiled Rust Claw harness."""
        print(f"\n--- Neuro is waking up the Claw Engineer... ---")
        
        # Path to the binary you compiled earlier
        claw_binary = os.path.expanduser("~/Project/NeuroClone/claw-code-main/rust/target/release/claw")
        
        if not os.path.exists(claw_binary):
            return "(Claw Engineer is offline. Binary not found.)"

        try:
            # We use the 'query' command from the Rust port to run the task
            result = subprocess.run(
                [claw_binary, "query", task],
                capture_output=True,
                text=True,
                timeout=60 # Give the Rust agent a minute to read your files
            )
            
            if result.returncode == 0:
                return f"\n[Claw Engineer Report]:\n{result.stdout}\n"
            else:
                return f"\n[Claw Engineer Error]:\n{result.stderr}\n"
                
        except subprocess.TimeoutExpired:
            return "(Claw Engineer took too long and was terminated.)"
        except Exception as e:
            return f"(Claw connection failed: {e})"

    # -------------------------------------------------------------------------
    # System Stats
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Web Search — now actually fetches page content via crawl4ai
    # -------------------------------------------------------------------------
    def get_web_urls(self, query: str, num_results: int = 5) -> list[str]:
        """Get URLs from DuckDuckGo, filtering out useless sites."""
        try:
            discard = ["youtube.com", "britannica.com", "vimeo.com", "instagram.com", "twitter.com"]
            clean_query = query
            for site in discard:
                clean_query += f" -site:{site}"
            results = DDGS().text(clean_query, max_results=num_results)
            return [r["href"] for r in results if "href" in r]
        except Exception as e:
            print(f"[DDG Error]: {e}")
            return []

    async def _crawl_urls(self, urls: list[str]) -> str:
        """Actually fetch and parse page content from URLs using crawl4ai."""
        content_filter = BM25ContentFilter(user_query="", use_stemming=False)
        md_generator = DefaultMarkdownGenerator(content_filter=content_filter)
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            markdown_generator=md_generator,
            page_timeout=8000,       # 8s per page max
            wait_until="domcontentloaded"
        )

        results_text = ""
        async with AsyncWebCrawler() as crawler:
            for url in urls[:3]:   # Only crawl top 3 to keep it fast
                try:
                    result = await crawler.arun(url=url, config=run_config)
                    if result.success and result.markdown:
                        # Trim to first 800 chars so we don't flood the context
                        snippet = result.markdown.fit_markdown[:800].strip()
                        results_text += f"\nSource: {url}\n{snippet}\n{'-'*40}\n"
                    else:
                        results_text += f"\nSource: {url}\n(Could not load page content)\n"
                except Exception as e:
                    results_text += f"\nSource: {url}\n(Crawl error: {e})\n"
        return results_text

    def search_web(self, query: str) -> str:
        """Full web search: DDG → crawl pages → return content."""
        print(f"--- Neuro is scanning the web for: '{query}' ---")
        urls = self.get_web_urls(query)
        if not urls:
            return "\n(Web search returned no results)\n"

        # Run the async crawler synchronously
        try:
            content = asyncio.run(self._crawl_urls(urls))
            return f"\n--- Web Search Results ---\n{content}\n"
        except Exception as e:
            return f"\n(Web crawl failed: {e})\n"

    # -------------------------------------------------------------------------
    # RAG Helpers
    # -------------------------------------------------------------------------
    def sync_knowledge(self) -> str:
        """Manually re-scan the knowledge/ folder and index new files."""
        if not self.rag:
            return "(RAG is offline — cannot sync knowledge.)"
        n = self.rag.sync_knowledge_folder()
        return f"(Synced knowledge folder — {n} new chunks indexed.)" if n else "(Knowledge folder is already up-to-date.)"

    # -------------------------------------------------------------------------
    # Main Think Loop
    # -------------------------------------------------------------------------
    def think(self, user_input: str) -> str:
        web_data = ""
        sys_stats = ""
        claw_data = ""
        rag_context = ""
        lower = user_input.lower()

        # 1. Trigger Claw Harness for code tasks
        if lower.startswith("claw:"):
            claw_task = user_input[5:].strip()
            claw_data = self.delegate_to_claw(claw_task)
            user_input = "Please briefly summarize what the Claw Engineer just found or did."

        # 2. Trigger web search
        elif any(word in lower for word in self.web_triggers):
            web_data = self.search_web(user_input)

        # 3. Trigger system stats
        elif any(word in lower for word in self.system_triggers):
            sys_stats = f"[Live System Stats: {self.get_system_stats()}]\n"

        # 4. Trigger explicit knowledge sync
        elif "sync knowledge" in lower or ("learn" in lower and "document" in lower):
            return self.sync_knowledge()

        # 5. RAG — retrieve context and inject into LLM prompt
        if self.rag:
            is_explicit_recall = any(t in lower for t in self.rag_triggers)
            try:
                hits = self.rag.retrieve_all(user_input)
                if hits:
                    # Build a clean context block from the hits
                    snippets = []
                    for h in hits[:4]:
                        src = h["metadata"].get("source", h["metadata"].get("type", "memory"))
                        snippets.append(f"[{src}]: {h['document'][:350]}")
                    rag_context = "\n".join(snippets)

                    # On explicit recall, add an instruction so Neuro responds conversationally
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

        # 6. Safely build the full prompt context
        context = ""
        if rag_context:
            context += rag_context + "\n\n"
        if claw_data:
            context += claw_data + "\n"
        if web_data:
            context += web_data + "\n"
        if sys_stats:
            context += sys_stats + "\n"

        final_prompt = f"{context}{user_input}" if context else user_input
        self.history.append({"role": "user", "content": final_prompt})

        payload = {
            "model": self.model,
            "messages": self.history[-11:],   # Rolling 11-message window
            "stream": False,
            "keep_alive": "5m",               # Unload after response (saves VRAM for RVC)
            "options": {
                "temperature": 0.8
            }
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

            self.history.append({"role": "assistant", "content": reply})

            # 7. Persist this turn to RAG memory (non-blocking best-effort)
            if self.rag:
                try:
                    self.rag.save_memory(user_input, reply)
                except Exception as e:
                    print(f"[RAG] Memory save error: {e}")

            return reply

        except requests.exceptions.ConnectionError:
            return "Neuro's brain is offline. Is Ollama running?"
        except requests.exceptions.Timeout:
            return "Neuro took too long to think. Try a shorter question."
        except requests.exceptions.RequestException as e:
            return f"Neuro's brain disconnected: {e}"
        except Exception as e:
            return f"Neuro is confused: {e}"
