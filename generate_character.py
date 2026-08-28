"""Cliente del pipeline persona->personaje.

  prompt --> Z-Image Turbo (Runware, ~$0.0013) --> face swap (RunPod serverless CPU) --> PNG final

Uso:
  python generate_character.py \
    --source foto.png \
    --prompt "cinematic film still of a male elf warrior like Legolas..." \
    --out final.png [--seed 333] [--width 832] [--height 1248]

Credenciales (variables de entorno o archivos):
  RUNWARE_API_KEY  o  ~/.keys/runware.key
  RUNPOD_API_KEY   o  ~/.keys/runpod.key
  RUNPOD_SWAP_ENDPOINT  (ej. https://api.runpod.ai/v2/<id>)  o  ~/.keys/runpod_swap_endpoint
"""
import argparse
import base64
import os
import sys
import time
import uuid

import requests

RUNWARE_URL = "https://api.runware.ai/v1"


def credential(env_name, file_name):
    v = os.environ.get(env_name)
    if v:
        return v.strip()
    path = os.path.expanduser(f"~/.keys/{file_name}")
    if os.path.exists(path):
        return open(path).read().strip()
    sys.exit(f"falta credencial: defini {env_name} o crea {path}")


def generate_base(prompt, seed, width, height, key):
    task = {
        "taskType": "imageInference", "taskUUID": str(uuid.uuid4()),
        "positivePrompt": prompt, "model": "runware:z-image@turbo",
        "width": width, "height": height, "numberResults": 1,
        "seed": seed, "includeCost": True,
    }
    r = requests.post(RUNWARE_URL, json=[task], timeout=180,
                      headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    body = r.json()
    if body.get("errors"):
        sys.exit(f"error de Runware: {body['errors']}")
    data = body["data"][0]
    print(f"[runware] imagen base lista (costo ${data.get('cost')})")
    return data["imageURL"]


def prepare_source(path):
    """Re-codifica la foto fuente sin metadatos, a resolucion COMPLETA.

    Sin exif_transpose (tags viciados; el worker resuelve orientacion por
    deteccion) y sin reducir: probado que bajar a 1200 o incluso 2000px
    degrada la identidad del swap (bigote manchado, rasgos lavados)."""
    import io

    from PIL import Image
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def swap_face(source_path, target_url, endpoint, key):
    src_b64 = base64.b64encode(prepare_source(source_path)).decode()
    payload = {"input": {
        "source_image": "data:image/png;base64," + src_b64,
        "target_image": target_url,
        "enhance": True,
    }}
    headers = {"Authorization": f"Bearer {key}"}
    r = requests.post(f"{endpoint}/runsync", json=payload, timeout=300, headers=headers)
    r.raise_for_status()
    job = r.json()
    while job.get("status") in ("IN_QUEUE", "IN_PROGRESS"):
        time.sleep(5)
        job = requests.get(f"{endpoint}/status/{job['id']}", timeout=30,
                           headers=headers).json()
    if job.get("status") != "COMPLETED":
        sys.exit(f"swap fallo: {job.get('status')} {job.get('output')}")
    out = job["output"]
    if "error" in out:
        sys.exit(f"swap error: {out['error']}")
    print(f"[runpod] swap listo (delay {job.get('delayTime')}ms, "
          f"ejecucion {job.get('executionTime')}ms)")
    return base64.b64decode(out["image"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="foto de la persona")
    ap.add_argument("--prompt", required=True, help="descripcion del personaje")
    ap.add_argument("--out", default="final.png")
    ap.add_argument("--seed", type=int, default=333)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=1248)
    args = ap.parse_args()

    runware_key = credential("RUNWARE_API_KEY", "runware.key")
    runpod_key = credential("RUNPOD_API_KEY", "runpod.key")
    endpoint = credential("RUNPOD_SWAP_ENDPOINT", "runpod_swap_endpoint")

    base_url = generate_base(args.prompt, args.seed, args.width, args.height, runware_key)
    png = swap_face(args.source, base_url, endpoint, runpod_key)
    open(args.out, "wb").write(png)
    print(f"[listo] {args.out}")


if __name__ == "__main__":
    main()
