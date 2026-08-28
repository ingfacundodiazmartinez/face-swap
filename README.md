# faceswap-worker (RunPod serverless, CPU)

Face swap (inswapper_128) + restauracion (GFPGAN v1.4) como worker serverless de RunPod,
pensado para CPU (sin GPU, sin colas por escasez). Modelos horneados en la imagen:
el arranque en frio no descarga nada.

## Pipeline completo

```
prompt -> Z-Image Turbo (Runware, ~$0.0013) -> este worker (~$0.001 en CPU) -> imagen final
```

Cliente de referencia: `generate_character.py`.

## Deploy en RunPod

1. New Endpoint -> Import from GitHub -> este repo, rama `runpod-worker`, Dockerfile en la raiz.
2. Worker type: **CPU** (4-8 vCPU alcanza). Scale to zero. Idle timeout: 60s.
3. Probar con:

```json
{"input": {"source_image": "<url o base64 de la cara>",
           "target_image": "<url o base64 del personaje>",
           "enhance": true}}
```

Respuesta: `{"output": {"image": "<png base64>"}}`

## Notas

- inswapper_128 es de uso NO comercial (licencia InsightFace). Para producto pago,
  evaluar GHOST/SimSwap (Apache) o asumir el riesgo.
- GFPGAN corre con `only_center_face=True` (evita duplicaciones en imagenes grandes).
- Tiempos medidos en CPU: ~5-10s por swap con worker caliente.
