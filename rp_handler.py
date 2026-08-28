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
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("no se pudo decodificar la imagen")
    return img


def best_face_any_orientation(img):
    """Detecta la cara probando las 4 orientaciones y devuelve la mejor.

    Los metadatos EXIF son poco confiables (tags viciados, librerias que los
    interpretan distinto); el puntaje del detector es la unica verdad."""
    rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180,
                 cv2.ROTATE_90_COUNTERCLOCKWISE]
    best_face, best_score = None, -1.0
    for rot in rotations:
        candidate = img if rot is None else cv2.rotate(img, rot)
        faces = app.get(candidate)
        if faces:
            f = biggest(faces)
            if f.det_score > best_score:
                best_face, best_score = f, float(f.det_score)
    return best_face


def biggest(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def handler(job):
    inp = job["input"]
    src = load_image(inp["source_image"])
    tgt = load_image(inp["target_image"])

    src_face = best_face_any_orientation(src)
    if src_face is None:
        return {"error": "no se detecto cara en source_image"}
    tgt_faces = app.get(tgt)
    if not tgt_faces:
        return {"error": "no se detecto cara en target_image"}

    out = swapper.get(tgt, biggest(tgt_faces), src_face, paste_back=True)
    if inp.get("enhance", True):
        _, _, out = enhancer.enhance(out, only_center_face=True, paste_back=True)

    ok, buf = cv2.imencode(".png", out)
    if not ok:
        return {"error": "fallo al codificar la salida"}
    return {"image": base64.b64encode(buf.tobytes()).decode()}


runpod.serverless.start({"handler": handler})
