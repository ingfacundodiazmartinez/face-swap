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


def adapt_prompt_to_gender(prompt, source_image):
    """Adapta la base al genero de la cara fuente (lo detecta buffalo_l).

    Convencion: el token {person} en el prompt se reemplaza por
    "young woman" / "young man". Sin token, se agrega una frase al final.
    Para caras femeninas ademas se elimina "clean shaven face": contradice
    y masculiniza la mandibula de la base."""
    try:
        face, _ = rp_handler.best_face_any_orientation(
            rp_handler.load_image(source_image))
        sex = getattr(face, "sex", None) if face is not None else None
    except Exception:
        sex = None
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
        prompt = adapt_prompt_to_gender(prompt, body["source_image"])
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
