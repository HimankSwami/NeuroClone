import os
import torch
import fairseq

# This permanently bypasses the PyTorch 2.6 security check for this script
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])

import subprocess
import os
import torch
import re
import fairseq

# Ensure PyTorch trusts the RVC model files
torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])

def preprocess_text(text):

    text = text.replace("*", "")  # Remove asterisks used for emphasis
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = re.sub(r'!+', '!', text)
    # 1. Turn actions in brackets into a 1.5 second silence
    # Using [[ . ]] tells Piper to insert a phoneme-level pause
    text = re.sub(r'\(.*?\)', r'[[ . ]] [[ . ]] [[ . ]]', text)

    # 2. Turn '...' into a 1 second hesitant pause
    text = text.replace("...", "[[ . ]] [[ . ]]")

    # 3. Add a slight breath after exclamation marks
    text = text.replace("!", "! [[ . ]]")

    return " ".join(text.split())

def speak(text):
    # --- Clear GPU Memory and Preprocess ---
    torch.cuda.empty_cache()
    processed_text = preprocess_text(text)
    
    # Escape double quotes for the shell command
    # We do this outside the f-string to avoid Python SyntaxErrors
    shell_safe_text = processed_text.replace('"', '\\"')
    
    base_path = os.path.expanduser("~/Project/NeuroClone")
    model_path = os.path.join(base_path, "models", "ayaka.pth")
    index_path = os.path.join(base_path, "models", "ayaka.index")
    temp_audio = os.path.join(base_path, "voice", "temp.wav")
    final_audio = os.path.join(base_path, "voice", "output.wav")

    # 1. Clean up
    if os.path.exists(temp_audio): os.remove(temp_audio)
    if os.path.exists(final_audio): os.remove(final_audio)

    # 2. Generate Base Voice (Piper)
    # Notice the use of shell_safe_text here
    piper_cmd = (
        f'echo "{shell_safe_text}" | python3 -m piper '
        f'--model {base_path}/models/en_US-hfc_female-medium.onnx '
        f'--output_file {temp_audio} --length_scale 0.85 --noise_scale 0.7 --noise_w 0.8'
    )
    
    try:
        print(f"Generating base voice with Piper...")
        subprocess.run(piper_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Piper Error: {e}")
        return

    # 3. Convert to Anime Voice via RVC CLI
    print(f"Applying Ayaka v2 Voice Conversion via CLI...")
    rvc_cmd = (
        f"python3 -m rvc_python cli -i {temp_audio} -o {final_audio} "
        f"-mp {model_path} -ip {index_path} -v v2 -pi 2 -me rmvpe -de cuda:0"
    )
    
    try:
        subprocess.run(rvc_cmd, shell=True, check=True)
    except Exception as e:
        print(f"RVC CLI Error: {e}")

    # 4. Play the result
    if os.path.exists(final_audio):
        os.system(f"aplay {final_audio}")
    else:
        print("RVC conversion failed, playing base Piper voice as backup.")
        os.system(f"aplay {temp_audio}")