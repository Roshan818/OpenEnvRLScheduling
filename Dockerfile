FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# LLM / inference config
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
ENV PORT=7860

# Factory task difficulty (easy | medium | hard)
ENV FACTORY_TASK=easy

# Enable built-in Gradio web UI at /web (with redirect from /)
ENV ENABLE_WEB_INTERFACE=1

EXPOSE 7860

CMD ["python", "server.py"]
