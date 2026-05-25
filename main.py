import sys
import speech_recognition as sr
from brain.core import NeuroBrain
from voice.speaker import speak

# -------------------------------------------------------------------------
# Voice Input
# -------------------------------------------------------------------------
def listen() -> str | None:
    """
    Listens to microphone and returns transcribed text.
    Returns:
        str   — transcribed text (confirmed by user)
        "RETRY" — user wants to re-record
        None  — unrecoverable error
    """
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 2.0   # Shorter than 5s — feels more natural

    with sr.Microphone() as source:
        print("\n--- Listening... (speak now) ---")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=None, phrase_time_limit=15)
            print("--- Processing speech... ---")
            text = recognizer.recognize_google(audio)

            print(f"\n  Captured: \"{text}\"")
            print("  [Enter] Send  |  [r] Retry  |  [Ctrl+C] Exit")

            while True:
                choice = input("  >> ").lower().strip()
                if choice == "":
                    return text
                elif choice == "r":
                    return "RETRY"
                else:
                    print("  Invalid. Press Enter to send or 'r' to retry.")

        except sr.UnknownValueError:
            print("  (Couldn't understand that, retrying...)")
            return "RETRY"
        except sr.RequestError as e:
            print(f"  [Speech API Error]: {e}")
            return None
        except Exception as e:
            print(f"  [Listen Error]: {e}")
            return None


# -------------------------------------------------------------------------
# Main Loop
# -------------------------------------------------------------------------
def run_neuro(voice_mode: bool = False):
    brain = NeuroBrain()
    mode = "voice" if voice_mode else "text"
    print(f"\n{'='*40}")
    print(f"  Neuro is online! (mode: {mode})")
    print(f"  Type 'exit' or press Ctrl+C to quit.")
    print(f"  Type 'voice' to enable speech, 'text' for text-only responses.")
    print(f"  Type 'rag stats' to check memory & knowledge status.")
    print(f"  Drop files into knowledge/ then say 'sync knowledge' to index them.")
    print(f"{'='*40}\n")

    speaks = voice_mode  # default: speak only if started in voice mode

    while True:
        try:
            # --- Get Input ---
            if voice_mode:
                user_input = listen()

                if user_input == "RETRY" or user_input is None:
                    continue  # Just re-listen, don't bother Neuro

            else:
                user_input = input("You: ").strip()

            # --- Sanity Checks ---
            if not user_input:
                continue

            lower = user_input.lower()

            if lower == "exit":
                print("Neuro: (waves) Later.")
                break

            # Toggle speech output
            if lower == "voice":
                speaks = True
                print("  [Voice responses enabled]")
                continue
            if lower == "text":
                speaks = False
                print("  [Text-only mode enabled]")
                continue
            if lower == "rag stats":
                if brain.rag:
                    s = brain.rag.stats()
                    print(f"  [RAG] Memories: {s['memory_count']}  |  Knowledge chunks: {s['knowledge_count']}")
                else:
                    print("  [RAG] Offline.")
                continue

            # --- Think & Respond ---
            print("  (Neuro is thinking...)")
            response = brain.think(user_input)

            # Strip <think>...</think> blocks if model outputs them (Qwen3 does this)
            import re
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

            print(f"\nNeuro: {response}\n")

            if speaks:
                speak(response)

        except KeyboardInterrupt:
            print("\nNeuro: (sighs) Fine. Shutting down.")
            break
        except Exception as e:
            print(f"[Run Error]: {e}")
            continue


if __name__ == "__main__":
    # Pass --voice flag to start in voice mode: python main.py --voice
    voice = "--voice" in sys.argv
    run_neuro(voice_mode=voice)
