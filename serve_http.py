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


@app.post("/runsync")
async def runsync(request: Request):
    token = os.environ.get("SWAP_TOKEN")
    if token:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="token invalido")

    body = await request.json()
    started = time.time()
    output = rp_handler.handler({"input": body.get("input", {})})
    elapsed_ms = int((time.time() - started) * 1000)

    status = "FAILED" if isinstance(output, dict) and "error" in output else "COMPLETED"
    return {"id": "fly", "status": status, "output": output,
            "delayTime": 0, "executionTime": elapsed_ms}
