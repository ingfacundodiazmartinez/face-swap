# Worker de face swap en CPU para RunPod serverless.
# Modelos horneados en la imagen: sin descargas en el arranque en frio.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ curl unzip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir runpod insightface==0.7.3 onnxruntime \
    opencv-python-headless gfpgan numpy pillow requests

# basicsr (dependencia de gfpgan) usa un import removido de torchvision
RUN sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
    /usr/local/lib/python3.11/site-packages/basicsr/data/degradations.py

# Modelos: inswapper, GFPGAN, detector buffalo_l, auxiliares de facexlib
# (rebuild v2: capa re-descargada tras corrupcion por crash del host)
RUN mkdir -p /app/models /root/.insightface/models/buffalo_l /app/gfpgan/weights && \
    curl -fsSL -o /app/models/inswapper_128.onnx \
      https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx && \
    curl -fsSL -o /app/models/GFPGANv1.4.pth \
      https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth && \
    curl -fsSL -o /tmp/buffalo_l.zip \
      https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && \
    unzip -q /tmp/buffalo_l.zip -d /root/.insightface/models/buffalo_l && rm /tmp/buffalo_l.zip && \
    curl -fsSL -o /app/gfpgan/weights/detection_Resnet50_Final.pth \
      https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth && \
    curl -fsSL -o /app/gfpgan/weights/parsing_parsenet.pth \
      https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth

COPY rp_handler.py /app/

CMD ["python", "-u", "rp_handler.py"]
