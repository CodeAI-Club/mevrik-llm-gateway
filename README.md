# LLM Gateway

OpenAI-compatible reverse proxy for multiple vLLM backends. Models are stored in `models.json` and can be added/removed
at runtime via API.

## Quick Start

### Docker

```bash
docker build -t llm-gateway .
docker run -d \
  --name llm-gateway \
  --env-file .env \
  -v $(pwd)/models.json:/app/models.json \
  -p 8000:8000 \
  --restart unless-stopped \
  llm-gateway
```

> Mount `models.json` so runtime changes persist across restarts.

### Without Docker

```bash
pip install -r requirements.txt
python run.py
```

## Managing Models

### Add a model at runtime

```bash
curl -X POST http://localhost:8000/v1/models \
  -H "Authorization: Bearer sk-your-secret-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "llama3-70b",
    "name": "Llama 3 70B",
    "backend_url": "http://10.10.121.68:8020/v1",
    "backend_model": "meta-llama/Llama-3-70B",
    "model_type": "chat",
    "owned_by": "meta"
  }'
```

### List models

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer sk-your-secret-key-change-me"
```

### Update a model

```bash
curl -X PATCH http://localhost:8000/v1/models/llama3-70b \
  -H "Authorization: Bearer sk-your-secret-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{"backend_url": "http://10.10.121.68:8025/v1"}'
```

### Delete a model

```bash
curl -X DELETE http://localhost:8000/v1/models/llama3-70b \
  -H "Authorization: Bearer sk-your-secret-key-change-me"
```

## Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://your-server:8000/v1",
    api_key="sk-your-secret-key-change-me",
)

# Chat
resp = client.chat.completions.create(
    model="qwen3.5-9b",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Streaming
for chunk in client.chat.completions.create(
        model="qwen3.5-9b",
        messages=[{"role": "user", "content": "Write a poem"}],
        stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")

# Embeddings
emb = client.embeddings.create(model="qwen3-embedding", input="Hello world")

# List models
for m in client.models.list().data:
    print(m.id)
```

## API Endpoints

| Method | Endpoint                   | Description                            |
|--------|----------------------------|----------------------------------------|
| GET    | `/health`                  | Health check                           |
| GET    | `/v1/models`               | List all models                        |
| GET    | `/v1/models/{id}`          | Get model info                         |
| POST   | `/v1/models`               | Add model (persists to models.json)    |
| PATCH  | `/v1/models/{id}`          | Update model fields                    |
| DELETE | `/v1/models/{id}`          | Remove model                           |
| POST   | `/v1/chat/completions`     | Chat (streaming supported)             |
| POST   | `/v1/completions`          | Text completions (streaming supported) |
| POST   | `/v1/embeddings`           | Embeddings                             |
| POST   | `/v1/rerank`               | Rerank documents                       |
| POST   | `/v1/score`                | Cross-encoder scoring                  |
| POST   | `/v1/models/{id}/test`     | Test model latency, TTFB, tokens/sec   |
| POST   | `/v1/models/{id}/loadtest` | Concurrent stress test                 |
| GET    | `/v1/stats`                | Per-user token speed & latency stats   |
| GET    | `/v1/stats/users`          | List all tracked user keys             |
| DELETE | `/v1/stats`                | Clear stats                            |

Interactive docs available at `/docs`.