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
    best_face, best_upright, best_idx = None, -2.0, -1
    uprightness, scores = [], []
    for idx, rot in enumerate(rotations):
        candidate = img if rot is None else cv2.rotate(img, rot)
        faces = app.get(candidate)
        if not faces:
            uprightness.append(-2.0)
            scores.append(0.0)
            continue
        f = biggest(faces)
        # Geometria manda: en la orientacion correcta los ojos estan ARRIBA
        # de la boca. El det_score NO sirve de oraculo (puntua caras de
        # costado por encima de la derecha; verificado con datos reales).
        eyes = (f.kps[0] + f.kps[1]) / 2.0
        mouth = (f.kps[3] + f.kps[4]) / 2.0
        v = mouth - eyes
        up = float(v[1] / (np.linalg.norm(v) + 1e-6))  # 1.0 = perfectamente derecha
        uprightness.append(round(up, 4))
        scores.append(round(float(f.det_score), 4))
        if up > best_upright:
            best_face, best_upright, best_idx = f, up, idx
    return best_face, {"rotacion_elegida": best_idx,
                       "verticalidad": uprightness, "puntajes": scores}


def biggest(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def handler(job):
    inp = job["input"]
    sources = inp.get("source_images") or [inp["source_image"]]
    tgt = load_image(inp["target_image"])

    faces, debug = [], []
    for value in sources:
        face, info = best_face_any_orientation(load_image(value))
        if face is not None:
            faces.append(face)
            debug.append(info)
    if not faces:
        return {"error": "no se detecto cara en ninguna source_image"}

    # Con varias fotos se promedia la identidad: diluye rarezas de una sola
    if len(faces) > 1:
        emb = np.mean([f.normed_embedding for f in faces], axis=0)
        emb = emb / np.linalg.norm(emb)
        src_face = faces[0]
        src_face.embedding = emb * np.linalg.norm(faces[0].embedding)
    else:
        src_face = faces[0]

    tgt_faces = app.get(tgt)
    if not tgt_faces:
        return {"error": "no se detecto cara en target_image"}

    out = swapper.get(tgt, biggest(tgt_faces), src_face, paste_back=True)
    if inp.get("enhance", True):
        _, _, out = enhancer.enhance(out, only_center_face=True, paste_back=True)

    ok, buf = cv2.imencode(".png", out)
    if not ok:
        return {"error": "fallo al codificar la salida"}
    result = {"image": base64.b64encode(buf.tobytes()).decode()}
    if inp.get("debug"):
        result["debug"] = {"fuentes": debug, "num_fuentes": len(faces)}
    return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
