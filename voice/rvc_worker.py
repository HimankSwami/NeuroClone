"""
rvc_worker.py — persistent RVC worker process.
Reads jobs from stdin as JSON lines, writes results to stdout.
Run once, keep alive — model stays loaded between calls.
"""
import sys, os, json, subprocess, tempfile

BASE_PATH  = os.path.expanduser("~/Project/NeuroClone")
MODEL_PATH = os.path.join(BASE_PATH, "models", "infamous_miku_v2.pth")
INDEX_PATH = os.path.join(BASE_PATH, "models", "added_IVF2983_Flat_nprobe_1_infamous_miku_v2_v2.index")

# Signal ready
print(json.dumps({"status": "ready"}), flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        job = json.loads(line)
        inp = job["input"]
        out = job["output"]

        cmd = (
            f"rvc infer "
            f"-m {MODEL_PATH} "
            f"-i {inp} "
            f"-o {out} "
            f"-if {INDEX_PATH} "
            f"-fm rmvpe "
            f"-fu 0 "
            f"-ir 0.75 "
            f"-fr 3 "
            f"-p 0.33"
        )
        result = subprocess.run(cmd, shell=True,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)
        if result.returncode == 0 and os.path.exists(out):
            print(json.dumps({"status": "done"}), flush=True)
        else:
            err = result.stderr.decode().strip()[-200:]
            print(json.dumps({"status": "error", "msg": err}), flush=True)
    except Exception as e:
        print(json.dumps({"status": "error", "msg": str(e)}), flush=True)