import os
import re
import subprocess
import torch
from rvc_python import RVC  # <-- use the package you already have

# Paths – keep yours exactly the same
BASE_PATH      = os.path.expanduser("~/Project/NeuroClone")
MODEL_PATH     = os.path.join(BASE_PATH, "models", "ayaka.pth")
INDEX_PATH     = os.path.join(BASE_PATH, "models", "ayaka.index")
PIPER_MODEL    = os.path.join(BASE_PATH, "models", "en_US-hfc_female-medium.onnx")
TEMP_AUDIO     = os.path.join(BASE_PATH, "voice", "temp.wav")
FINAL_AUDIO    = os.path.join(BASE_PATH, "voice", "output.wav")

# -------------------------------------------------------------------------
# Persistent Voice Engine — loads RVC once and keeps it alive in VRAM
# -------------------------------------------------------------------------
class VoiceEngine:
    def __init__(self):
        print("[Voice] Loading RVC model into GPU VRAM...")
        # Create RVC instance with GPU
        self.rvc = RVC(device="cuda")
        # Load the model + index (this stays in memory)
        self.rvc.load_model(MODEL_PATH, INDEX_PATH)
        print("[Voice] RVC model ready. VRAM is now your friend.")

    def preprocess_text(self, text: str) -> str:
        text = text.replace("*", "")
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        text = re.sub(r'!+', '!', text)
        text = re.sub(r'\(.*?\)', '[[ . ]] [[ . ]] [[ . ]]', text)
        text = text.replace("...", "[[ . ]] [[ . ]]")
        text = text.replace("!", "! [[ . ]]")
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def speak(self, text: str) -> None:
        if not text or not text.strip():
            return

        torch.cuda.empty_cache()
        processed = self.preprocess_text(text)
        shell_safe = processed.replace('"', '\\"')

        # Clean up old files
        for path in [TEMP_AUDIO, FINAL_AUDIO]:
            if os.path.exists(path):
                os.remove(path)

        # ------------------------------------------------------------------
        # Step 1: Piper TTS (still CPU, but fast)
        # ------------------------------------------------------------------
        piper_cmd = (
            f'echo "{shell_safe}" | python3 -m piper '
            f'--model {PIPER_MODEL} '
            f'--output_file {TEMP_AUDIO} '
            f'--length_scale 0.85 --noise_scale 0.7 --noise_w 0.8'
        )

        try:
            print("  [Voice] Piper TTS...")
            subprocess.run(piper_cmd, shell=True, check=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"  [Piper Error]: {e}")
            return

        if not os.path.exists(TEMP_AUDIO):
            print("  [Voice] Piper failed.")
            return

        # ------------------------------------------------------------------
        # Step 2: RVC Conversion — runs on GPU instantly
        # (No subprocess, no reloading — just inference)
        # ------------------------------------------------------------------
        print("  [Voice] RVC converting on GPU...")
        try:
            # The rvc_python API usually has a method like convert()
            self.rvc.convert(
                input_path=TEMP_AUDIO,
                output_path=FINAL_AUDIO,
                pitch=0,
                f0_method='rmvpe'   # or 'crepe' – keep what you used before
            )
        except Exception as e:
            print(f"  [RVC Error]: {e}")
            print("  [Voice] Falling back to Piper...")
            self._play(TEMP_AUDIO)
            return

        # ------------------------------------------------------------------
        # Step 3: Play
        # ------------------------------------------------------------------
        if os.path.exists(FINAL_AUDIO):
            print("  [Voice] Playing...")
            self._play(FINAL_AUDIO)
        else:
            print("  [Voice] RVC output missing. Playing Piper.")
            self._play(TEMP_AUDIO)

    @staticmethod
    def _play(file_path: str) -> None:
        FLAG = os.path.join(BASE_PATH, "voice", "playing.flag")
        open(FLAG, "w").close()
        os.system(f"aplay {file_path}")
        if os.path.exists(FLAG):
            os.remove(FLAG)


# -------------------------------------------------------------------------
# Create ONE global instance that stays alive forever
# -------------------------------------------------------------------------
voice_engine = VoiceEngine()

# Keep your old 'speak' function name for backward compatibility
def speak(text: str) -> None:
    voice_engine.speak(text)