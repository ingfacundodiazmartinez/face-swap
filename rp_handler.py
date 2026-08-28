"""RunPod serverless worker: face swap (inswapper) + restauracion (GFPGAN), CPU.

Input:
  source_image: URL o base64 de la foto con la cara a usar
  target_image: URL o base64 de la imagen destino (el personaje)
  enhance: bool, restaurar con GFPGAN (default true)
Output:
  {"image": "<png base64>"}
"""
import base64

import cv2
import gfpgan
import insightface
import numpy as np
import requests
import runpod
from insightface.app import FaceAnalysis

print("[worker] cargando modelos...", flush=True)
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
swapper = insightface.model_zoo.get_model(
    "/app/models/inswapper_128.onnx", providers=["CPUExecutionProvider"])
enhancer = gfpgan.GFPGANer(model_path="/app/models/GFPGANv1.4.pth", upscale=1)
print("[worker] modelos listos", flush=True)


def load_image(value):
    if value.startswith("http"):
        data = requests.get(value, timeout=60).content
    else:
        data = base64.b64decode(value.split(",")[-1])
    # PIL aplica la orientacion EXIF de forma canonica y la descarta;
    # cv2.imdecode/imread la interpretan distinto segun version (bug real).
    import io
    from PIL import Image, ImageOps
    pil = Image.open(io.BytesIO(data))
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def biggest(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def handler(job):
    inp = job["input"]
    src = load_image(inp["source_image"])
    tgt = load_image(inp["target_image"])

    src_faces = app.get(src)
    if not src_faces:
        return {"error": "no se detecto cara en source_image"}
    tgt_faces = app.get(tgt)
    if not tgt_faces:
        return {"error": "no se detecto cara en target_image"}

    out = swapper.get(tgt, biggest(tgt_faces), biggest(src_faces), paste_back=True)
    if inp.get("enhance", True):
        _, _, out = enhancer.enhance(out, only_center_face=True, paste_back=True)

    ok, buf = cv2.imencode(".png", out)
    if not ok:
        return {"error": "fallo al codificar la salida"}
    return {"image": base64.b64encode(buf.tobytes()).decode()}


runpod.serverless.start({"handler": handler})
