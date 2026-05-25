"""
brain/core.py  —  PATCHED for RAG
===================================
This file shows exactly what to add/change in your existing core.py.
Look for the  ← ADD  and  ← CHANGE  comments.

SUMMARY OF CHANGES:
  1. Import NeuroRAG at the top
  2. Instantiate it inside NeuroBrain.__init__
  3. In your main respond() / chat() method:
       a. Build a RAG context block from the user message
       b. Inject it into the system prompt (or as a prefixed user turn)
       c. After getting the assistant reply, call rag.save_memory()
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1.  ADD this import near the top of brain/core.py
# ─────────────────────────────────────────────────────────────────────────────
from rag.rag_engine import NeuroRAG          # ← ADD


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Inside NeuroBrain.__init__  — ADD one line
# ─────────────────────────────────────────────────────────────────────────────
class NeuroBrain:
    def __init__(self, ...):
        # ... your existing init code ...

        self.rag = NeuroRAG()               # ← ADD  (starts ChromaDB, syncs knowledge/)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Inside your respond() / chat() / generate_reply() method
#     (whatever your main LLM-call function is named)
# ─────────────────────────────────────────────────────────────────────────────
    def respond(self, user_message: str) -> str:
        # ── RAG: retrieve relevant context ───────────────────────────────
        context_block = self.rag.build_context_block(user_message)   # ← ADD

        # ── Build the prompt ─────────────────────────────────────────────
        # OPTION A — inject as a system-level prefix (recommended):
        #   Prepend context_block to your existing system prompt string.
        #
        #   Before (example):
        #       system_prompt = "You are Neuro, a helpful AI assistant..."
        #
        #   After:
        system_prompt = self._base_system_prompt                       # ← CHANGE (keep your existing prompt)
        if context_block:
            system_prompt = context_block + "\n\n" + system_prompt     # ← ADD

        # OPTION B — inject as an extra user turn before the real message:
        #   messages = [
        #       {"role": "user", "content": context_block},
        #       {"role": "assistant", "content": "Understood, I'll use this context."},
        #       {"role": "user", "content": user_message},
        #   ]

        # ── Call Ollama (your existing code, unchanged) ───────────────────
        response = ollama.chat(
            model=self.model,
            messages=self.history + [{"role": "user", "content": user_message}],
            system=system_prompt,            # ← CHANGE  (was self._base_system_prompt)
        )
        assistant_reply = response["message"]["content"]

        # ── RAG: persist this turn to memory ─────────────────────────────
        self.rag.save_memory(user_message, assistant_reply)           # ← ADD

        return assistant_reply


# ─────────────────────────────────────────────────────────────────────────────
# 4.  (Optional) Expose a helper so main.py can trigger a manual re-sync
#     e.g. when the user says "hey neuro, learn from my documents"
# ─────────────────────────────────────────────────────────────────────────────
    def sync_knowledge(self) -> str:
        n = self.rag.sync_knowledge_folder()                          # ← ADD
        return f"Synced knowledge folder — {n} new chunks indexed."


# ─────────────────────────────────────────────────────────────────────────────
# EXISTING TRIGGER KEYWORD  (add to your keyword routing block)
# ─────────────────────────────────────────────────────────────────────────────
# In whichever block handles "search", "cpu", etc., add:
#
#   elif any(kw in user_message.lower() for kw in ["remember", "you know", "recall"]):
#       hits = self.rag.retrieve_all(user_message)
#       if hits:
#           return "Here's what I remember:\n" + "\n".join(h["document"] for h in hits[:3])
#       return "I don't have anything stored about that yet."
#
#   elif "learn" in user_message.lower() and "document" in user_message.lower():
#       return self.sync_knowledge()
