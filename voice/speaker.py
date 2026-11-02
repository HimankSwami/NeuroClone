import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ.setdefault("rmvpe_root",
    os.path.expanduser("~/miniconda3/envs/neuro/lib/python3.10/site-packages/rvc_python/base_model"))

import re
import subprocess
import threading
from pathlib import Path
from scipy.io import wavfile

BASE_PATH   = os.path.expanduser("~/Project/NeuroClone")
MODEL_PATH  = os.path.join(BASE_PATH, "models", "march 7th.pth")
INDEX_PATH  = os.path.join(BASE_PATH, "models", "added_IVF205_Flat_nprobe_1_march 7th_v2.index")
PIPER_MODEL = os.path.join(BASE_PATH, "models", "en_US-hfc_female-medium.onnx")
TEMP_AUDIO  = os.path.join(BASE_PATH, "voice", "temp.wav")
FINAL_AUDIO = os.path.join(BASE_PATH, "voice", "output.wav")
HUBERT_PATH = os.path.expanduser(
    "~/miniconda3/envs/neuro/lib/python3.10/site-packages/rvc_python/base_model/hubert_base.pt"
)
FLAG_PATH   = os.path.join(BASE_PATH, "voice", "playing.flag")

# ---------------------------------------------------------------------------
# Load RVC once — stays in memory forever
# ---------------------------------------------------------------------------
print("[Voice] Loading RVC model into memory...")
import torch
import fairseq
torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])
from rvc.modules.vc.modules import VC

# rvc lib needs these env vars
os.environ.setdefault("index_root", os.path.join(BASE_PATH, "models"))
os.environ.setdefault("weight_root", os.path.join(BASE_PATH, "models"))
os.environ.setdefault("outside_index_root", os.path.join(BASE_PATH, "models"))

_vc = VC()
_vc.get_vc(MODEL_PATH)
print("[Voice] RVC ready.")

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    text = text.replace("*", "")
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = re.sub(r'!+', '!', text)
    text = re.sub(r'\(.*?\)', '[[ . ]] [[ . ]] [[ . ]]', text)
    text = text.replace("...", "[[ . ]] [[ . ]]")
    text = text.replace("!", "! [[ . ]]")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _play(path: str):
    open(FLAG_PATH, "w").close()
    os.system(f"aplay {path}")
    if os.path.exists(FLAG_PATH):
        os.remove(FLAG_PATH)

# ---------------------------------------------------------------------------
# Main speak function
# ---------------------------------------------------------------------------
def speak(text: str) -> None:
    if not text or not text.strip():
        return

    processed = preprocess_text(text)
    shell_safe = processed.replace('"', '\\"')

    for path in [TEMP_AUDIO, FINAL_AUDIO]:
        if os.path.exists(path):
            os.remove(path)

    # Step 1: Piper TTS
    piper_cmd = (
        f'echo "{shell_safe}" | python3 -m piper '
        f'--model {PIPER_MODEL} '
        f'--output_file {TEMP_AUDIO} '
        f'--length_scale 0.99 --noise_scale 0.7 --noise_w 0.8'
    )
    try:
        print("  [Voice] Piper TTS...")
        subprocess.run(piper_cmd, shell=True, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"  [Piper Error]: {e.stderr.decode().strip()}")
        return

    if not os.path.exists(TEMP_AUDIO):
        print("  [Voice] Piper failed.")
        return

    # Step 2: RVC — model already loaded, just run inference
    print("  [Voice] RVC converting...")
    try:
        tgt_sr, audio_opt, times, _ = _vc.vc_single(
            sid=0,
            input_audio_path=Path(TEMP_AUDIO),
            f0_up_key=5,
            f0_method="rmvpe",
            f0_file=None,
            index_file=str(INDEX_PATH),   # Fix 1: str() not Path()
            index_rate=0.95,
            filter_radius=3,
            resample_sr=0,
            rms_mix_rate=0.25,
            protect=0.33,
            hubert_path=HUBERT_PATH,
            
        )
        wavfile.write(FINAL_AUDIO, tgt_sr, audio_opt)
        print(f"  [Voice] RVC done in {times}")
    except Exception as e:
        print(f"  [RVC Error]: {e}")
        print("  [Voice] Falling back to Piper...")
        _play(TEMP_AUDIO)
        return

    # Step 3: Play
    if os.path.exists(FINAL_AUDIO):
        print("  [Voice] Playing...")
        _play(FINAL_AUDIO)
    else:
        print("  [Voice] Output missing — Piper fallback...")
        _play(TEMP_AUDIO)