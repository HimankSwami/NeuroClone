import os
import re
import subprocess
import torch
import fairseq

# Bypass PyTorch 2.6 security check for RVC model files
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])

# -------------------------------------------------------------------------
# Paths — edit BASE_PATH if your project lives elsewhere
# -------------------------------------------------------------------------
BASE_PATH      = os.path.expanduser("~/Project/NeuroClone")
MODEL_PATH     = os.path.join(BASE_PATH, "models", "ayaka.pth")
INDEX_PATH     = os.path.join(BASE_PATH, "models", "ayaka.index")
PIPER_MODEL    = os.path.join(BASE_PATH, "models", "en_US-hfc_female-medium.onnx")
TEMP_AUDIO     = os.path.join(BASE_PATH, "voice", "temp.wav")
FINAL_AUDIO    = os.path.join(BASE_PATH, "voice", "output.wav")


# -------------------------------------------------------------------------
# Text Preprocessing — makes TTS sound more natural
# -------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    text = text.replace("*", "")                          # Remove markdown bold/italic
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)  # Remove emoji / surrogate chars
    text = re.sub(r'!+', '!', text)                       # Collapse multiple !!!
    text = re.sub(r'\(.*?\)', '[[ . ]] [[ . ]] [[ . ]]', text)  # (actions) → pause
    text = text.replace("...", "[[ . ]] [[ . ]]")         # ... → hesitation pause
    text = text.replace("!", "! [[ . ]]")                 # ! → slight breath after
    text = re.sub(r'\s+', ' ', text).strip()              # Normalise whitespace
    return text


# -------------------------------------------------------------------------
# Speak — Piper TTS → RVC voice conversion → aplay
# -------------------------------------------------------------------------
def speak(text: str) -> None:
    if not text or not text.strip():
        return

    torch.cuda.empty_cache()
    processed = preprocess_text(text)

    # Escape any double quotes so the shell doesn't choke
    shell_safe = processed.replace('"', '\\"')

    # Clean up previous audio files
    for path in [TEMP_AUDIO, FINAL_AUDIO]:
        if os.path.exists(path):
            os.remove(path)

    # ------------------------------------------------------------------
    # Step 1: Generate base voice with Piper TTS
    # ------------------------------------------------------------------
    piper_cmd = (
        f'echo "{shell_safe}" | python3 -m piper '
        f'--model {PIPER_MODEL} '
        f'--output_file {TEMP_AUDIO} '
        f'--length_scale 0.85 --noise_scale 0.7 --noise_w 0.8'
    )

    try:
        print("  [TTS] Generating base voice...")
        subprocess.run(
            piper_cmd, shell=True, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(f"  [Piper Error]: {e.stderr.decode().strip()}")
        return

    if not os.path.exists(TEMP_AUDIO):
        print("  [Piper Error]: temp.wav was not created.")
        return

    # ------------------------------------------------------------------
    # Step 2: Convert to Ayaka voice via RVC CLI
    # ------------------------------------------------------------------
    rvc_cmd = (
        f"python3 -m rvc_python cli "
        f"-i {TEMP_AUDIO} -o {FINAL_AUDIO} "
        f"-mp {MODEL_PATH} -ip {INDEX_PATH} "
        f"-v v2 -pi 2 -me rmvpe -de cuda:0"
    )

    try:
        print("  [RVC] Applying Ayaka voice conversion...")
        subprocess.run(
            rvc_cmd, shell=True, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(f"  [RVC Error]: {e.stderr.decode().strip()}")
        print("  [Fallback] Playing base Piper voice...")
        os.system(f"aplay {TEMP_AUDIO}")
        return

    # ------------------------------------------------------------------
    # Step 3: Play the final converted audio
    # ------------------------------------------------------------------
    if os.path.exists(FINAL_AUDIO):
        print("  [Audio] Playing...")
        os.system(f"aplay {FINAL_AUDIO}")
    else:
        print("  [RVC] output.wav missing, falling back to Piper voice.")
        os.system(f"aplay {TEMP_AUDIO}")
