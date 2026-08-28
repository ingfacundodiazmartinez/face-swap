"""Servidor HTTP del face swap para Fly.io (o cualquier host).

Expone POST /runsync con el MISMO contrato que RunPod serverless, asi el
cliente funciona contra Fly o RunPod cambiando solo la URL.

Auth opcional: si la variable de entorno SWAP_TOKEN esta definida, el header
Authorization debe ser "Bearer <SWAP_TOKEN>".
"""
import os
import time

from fastapi import FastAPI, HTTPException, Request

import rp_handler  # carga los modelos una sola vez al arrancar el proceso

app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True}


def check_auth(request: Request):
    token = os.environ.get("SWAP_TOKEN")
    if token:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="token invalido")


@app.post("/runsync")
async def runsync(request: Request):
    check_auth(request)
    body = await request.json()
    started = time.time()
    output = rp_handler.handler({"input": body.get("input", {})})
    elapsed_ms = int((time.time() - started) * 1000)

    status = "FAILED" if isinstance(output, dict) and "error" in output else "COMPLETED"
    return {"id": "fly", "status": status, "output": output,
            "delayTime": 0, "executionTime": elapsed_ms}


def estimate_hair_color(img, face, debug):
    """Color de pelo aproximado muestreando la franja sobre la frente.

    Devuelve None si no hay pelo distinguible (calvicie, fondo colado):
    la mediana de esa franja se compara con la piel de la cara — si son
    parecidas, mejor no afirmar nada."""
    import numpy as np

    rotations = [None, None, None, None]
    try:
        import cv2
        rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180,
                     cv2.ROTATE_90_COUNTERCLOCKWISE]
        idx = debug.get("rotacion_elegida", 0)
        if rotations[idx] is not None:
            img = cv2.rotate(img, rotations[idx])
    except Exception:
        return None

    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    h = y2 - y1
    if h <= 0 or x2 <= x1:
        return None

    # Fondo: esquinas superiores de la imagen. Piel: centro de la cara.
    # La franja de pelo va pegada al nacimiento (y1-0.20h a y1+0.05h, centro
    # 60% en x) y se descartan los pixeles parecidos al fondo o a la piel —
    # sin esto, una pared clara detras da "gray" (bug real).
    fondo = np.median(np.concatenate([
        img[:max(1, H // 10), :max(1, W // 10)].reshape(-1, 3),
        img[:max(1, H // 10), -max(1, W // 10):].reshape(-1, 3)]), axis=0)
    cara = img[y1 + int(0.3 * h):y1 + int(0.7 * h), max(0, x1):x2]
    if cara.size == 0:
        return None
    piel = np.median(cara.reshape(-1, 3), axis=0)

    cx1 = max(0, x1 + int(0.2 * (x2 - x1)))
    cx2 = x2 - int(0.2 * (x2 - x1))
    band = img[max(0, y1 - int(0.20 * h)):y1 + int(0.05 * h), cx1:cx2]
    if band.size == 0:
        return None
    band = band.reshape(-1, 3).astype(float)

    pelo_px = band[(np.abs(band - fondo).sum(axis=1) > 120) &
                   (np.abs(band - piel).sum(axis=1) > 90)]
    if len(pelo_px) < 0.10 * len(band):  # calvicie o franja sin pelo
        return None

    ref = np.median(pelo_px, axis=0)  # BGR de cv2
    b, g, r = ref
    lum = 0.114 * b + 0.587 * g + 0.299 * r
    if lum < 45:
        color = "black"
    elif lum > 130 and abs(r - g) < 35 and abs(g - b) < 35 and abs(r - b) < 35:
        color = "gray"
    elif r > g > b and (r - b) > 45:
        color = "blond" if lum > 135 else ("auburn" if (r - g) > 35 else "brown")
    elif lum < 100:
        color = "dark brown"
    else:
        color = "brown"

    # Largo: el pelo largo flanquea la cara (orejas y mandibula). Calibrado
    # con /analyze sobre caras reales: largo da 0.8+, corto da <0.2.
    # Limite conocido: pelo largo atado atras lee "corto" (dano menor).
    fracciones = []
    for ya, yb, xa, xb in (
            (y1 + int(0.35 * h), y1 + int(0.65 * h), x1 - int(0.15 * w), x1),
            (y1 + int(0.35 * h), y1 + int(0.65 * h), x2, x2 + int(0.15 * w)),
            (y1 + int(0.55 * h), y1 + int(0.95 * h), x1 - int(0.15 * w), x1),
            (y1 + int(0.55 * h), y1 + int(0.95 * h), x2, x2 + int(0.15 * w))):
        zona = img[max(0, ya):min(H, yb), max(0, xa):min(W, xb)]
        if zona.size:
            zona = zona.reshape(-1, 3).astype(float)
            fracciones.append(float((np.abs(zona - ref).sum(axis=1) < 100).mean()))
    largo = "long" if fracciones and np.mean(fracciones) > 0.5 else "short"
    return f"{largo} {color}"


def describe_source_gemini(source_image):
    """Descripcion de la persona hecha por Gemini 2.5 Flash Lite (~$0.0001).

    Devuelve una frase lista para prompt ("man in his early thirties with
    short dark brown receding hair and light stubble") o None si no hay
    key o la llamada falla — en ese caso rigen las heuristicas locales."""
    import base64 as b64mod

    import requests as rq
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    try:
        if source_image.startswith("http"):
            raw = rq.get(source_image, timeout=30).content
            mime = "image/jpeg"
        else:
            cabeza, _, datos = source_image.partition(",")
            raw = b64mod.b64decode(datos or source_image)
            mime = "image/png" if "png" in cabeza else "image/jpeg"

        instruccion = (
            "Describe the main person in ONE short English phrase suitable for "
            "an image generation prompt. Start with 'man' or 'woman'. Include: "
            "apparent age range, hair color, hair length and style (mention "
            "receding hairline or baldness if present), facial hair if any, "
            "glasses if any. No names, no emotions, no clothing, no background. "
            "Example: 'man in his early thirties with short dark brown hair, "
            "receding hairline and light stubble'. Reply with the phrase only.")
        r = rq.post(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-2.5-flash-lite:generateContent",
            params={"key": key}, timeout=25,
            json={"contents": [{"parts": [
                {"inline_data": {"mime_type": mime,
                                 "data": b64mod.b64encode(raw).decode()}},
                {"text": instruccion}]}]})
        r.raise_for_status()
        texto = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        texto = " ".join(texto.split()).strip().strip(".").strip()
        if 10 < len(texto) < 250:
            print(f"[gemini] descripcion: {texto}", flush=True)
            return texto
    except Exception as error:
        print(f"[gemini] fallo ({error}); heuristicas locales", flush=True)
    return None


def adapt_prompt_to_source(prompt, source_image):
    """Adapta la base a la cara fuente: genero (buffalo_l lo detecta) y
    color de pelo (muestreo de pixeles).

    Convenciones en el basePrompt:
      {person} → "young woman" / "young man"
      "with {hair} hair" → color estimado, o se elimina la frase entera si
                           no se pudo estimar (calvicie, foto rara)
    Para caras femeninas ademas se elimina "clean shaven face": contradice
    y masculiniza la mandibula de la base."""
    # Camino preferido: descripcion de Gemini (cubre genero, pelo, barba,
    # lentes, edad — mejor que cualquier heuristica de pixeles).
    desc = describe_source_gemini(source_image)
    if desc:
        for frase in ("with {hair} hair, ", ", with {hair} hair",
                      "with {hair} hair", "with slicked {hair} hair"):
            prompt = prompt.replace(frase, "")
        if "{person}" in prompt:
            prompt = prompt.replace("{person}", desc)
        else:
            prompt += f". The person is a {desc}."
        # La descripcion manda sobre el vello facial del prompt fijo.
        for frase in ("clean shaven face, ", ", clean shaven face",
                      "clean shaven young face, ", "clean shaven face",
                      "clean shaven young face"):
            prompt = prompt.replace(frase, "")
        return prompt

    sex, hair = None, None
    try:
        img = rp_handler.load_image(source_image)
        face, debug = rp_handler.best_face_any_orientation(img)
        if face is not None:
            sex = getattr(face, "sex", None)
            hair = estimate_hair_color(img, face, debug)
    except Exception:
        pass

    if "{hair}" in prompt:
        if hair:
            prompt = prompt.replace("{hair}", hair)
        else:
            for frase in ("with {hair} hair, ", ", with {hair} hair",
                          "with {hair} hair"):
                prompt = prompt.replace(frase, "")

    if sex not in ("F", "M"):
        return prompt.replace("{person}", "young adult")

    word = "young woman" if sex == "F" else "young man"
    if "{person}" in prompt:
        prompt = prompt.replace("{person}", word)
    else:
        prompt += f". The character is a {word}."
    if sex == "F":
        for frase in ("clean shaven face, ", ", clean shaven face",
                      "clean shaven face"):
            prompt = prompt.replace(frase, "")
    return prompt


def generate_base_runware(prompt, seed, width, height):
    import uuid

    import requests as rq
    key = os.environ.get("RUNWARE_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="RUNWARE_API_KEY no configurada")
    task = {"taskType": "imageInference", "taskUUID": str(uuid.uuid4()),
            "positivePrompt": prompt, "model": "runware:z-image@turbo",
            "width": width, "height": height, "numberResults": 1, "seed": seed}
    r = rq.post("https://api.runware.ai/v1", json=[task], timeout=120,
                headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    body = r.json()
    if body.get("errors"):
        raise HTTPException(status_code=502, detail=f"runware: {body['errors']}")
    return body["data"][0]["imageURL"]


@app.post("/analyze")
async def analyze_source(request: Request):
    """Diagnostico: mediciones crudas de la cara fuente para calibrar las
    heuristicas de pelo (color y largo) con la bbox REAL del detector."""
    import cv2
    import numpy as np

    check_auth(request)
    body = await request.json()
    img = rp_handler.load_image(body["source_image"])
    face, debug = rp_handler.best_face_any_orientation(img)
    if face is None:
        raise HTTPException(status_code=422, detail="sin cara")
    rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180,
                 cv2.ROTATE_90_COUNTERCLOCKWISE]
    rot = rotations[debug.get("rotacion_elegida", 0)]
    if rot is not None:
        img = cv2.rotate(img, rot)

    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    h, w = y2 - y1, x2 - x1
    hair = estimate_hair_color(img, face, debug)

    def fraccion(ya, yb, xa, xb, ref):
        band = img[max(0, ya):min(H, yb), max(0, xa):min(W, xb)]
        if band.size == 0:
            return None
        band = band.reshape(-1, 3).astype(float)
        return round(float((np.abs(band - ref).sum(axis=1) < 100).mean()), 3)

    # color de pelo de referencia: mediana de la franja del nacimiento
    band_pelo = img[max(0, y1 - int(0.20 * h)):y1 + int(0.05 * h),
                    x1 + int(0.2 * w):x2 - int(0.2 * w)]
    ref = np.median(band_pelo.reshape(-1, 3).astype(float), axis=0) if band_pelo.size else None
    zonas = {}
    if ref is not None:
        for nombre, (ya, yb, xa, xb) in {
            "orejas_izq": (y1 + int(0.35 * h), y1 + int(0.65 * h), x1 - int(0.15 * w), x1),
            "orejas_der": (y1 + int(0.35 * h), y1 + int(0.65 * h), x2, x2 + int(0.15 * w)),
            "mandibula_izq": (y1 + int(0.55 * h), y1 + int(0.95 * h), x1 - int(0.15 * w), x1),
            "mandibula_der": (y1 + int(0.55 * h), y1 + int(0.95 * h), x2, x2 + int(0.15 * w)),
            "cuello_izq": (y2, y2 + int(0.4 * h), x1 - int(0.10 * w), x1 + int(0.15 * w)),
            "cuello_der": (y2, y2 + int(0.4 * h), x2 - int(0.15 * w), x2 + int(0.10 * w)),
        }.items():
            zonas[nombre] = fraccion(ya, yb, xa, xb, ref)

    return {"bbox": [x1, y1, x2, y2], "imagen": [W, H],
            "sexo": getattr(face, "sex", None), "color_pelo": hair,
            "ref_pelo_bgr": [round(float(v), 1) for v in ref] if ref is not None else None,
            "zonas": zonas, "debug": debug}


@app.post("/filter")
async def filter_ai(request: Request):
    """Filtro AI completo para talia: foto del usuario -> personaje.

    Body:
      source_image: URL o base64 de la cara del usuario (obligatorio)
      target_image: URL o base64 de la imagen base/personaje (opcional)
      prompt: si no hay target_image, el servidor genera la base con Z-Image
      seed, width, height: opcionales para la generacion (default 333, 832x1248)
      enhance: restaurar con GFPGAN (default true)
    Respuesta: {"image": <png b64>, "base_url": <url si se genero>, "executionTime": ms}
    """
    check_auth(request)
    body = await request.json()
    if "source_image" not in body:
        raise HTTPException(status_code=400, detail="falta source_image")

    started = time.time()
    base_url = None
    target = body.get("target_image")
    if not target:
        prompt = body.get("prompt")
        if not prompt:
            raise HTTPException(status_code=400,
                                detail="falta target_image o prompt")
        prompt = adapt_prompt_to_source(prompt, body["source_image"])
        base_url = generate_base_runware(prompt, int(body.get("seed", 333)),
                                         int(body.get("width", 832)),
                                         int(body.get("height", 1248)))
        target = base_url

    output = rp_handler.handler({"input": {
        "source_image": body["source_image"],
        "target_image": target,
        "enhance": body.get("enhance", True)}})
    if "error" in output:
        raise HTTPException(status_code=422, detail=output["error"])
    return {"image": output["image"], "base_url": base_url,
            "executionTime": int((time.time() - started) * 1000)}
