# CIMPLE Factors Server

A standalone HTTP API server for predicting emotion, sentiment, political leaning, narrative tropes, conspiracy factors, persuasion techniques, climate relatedness, and climate stance using BERT and T5-based models.

## 📋 Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (for dependency management)
- [just](https://github.com/casey/just) (for task automation)
- [Docker & Docker Compose](https://docs.docker.com/get-docker/) (for Docker setup)

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone the repository**:

```bash
git clone https://github.com/climatesense-project/cimple-factors-server.git
cd cimple-factors-server/
docker compose up --build
```

2. **Access the API**:

- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Local Development

1. **Setup environment**:

```bash
just setup-dev
```

2. **Run the server**:

```bash
just run
```

## API Usage

All model predictions are served through a single endpoint, `/predict/{model}`, where `{model}` is one of the servable models listed by `GET /models` (also returned in the `models` field of the `/models` response):

| Model                   | Output                                                    |
| ----------------------- | --------------------------------------------------------- |
| `emotion`               | Emotion label (Happiness, Anger, Sadness, Fear) or `null` |
| `sentiment`             | Negative / Neutral / Positive                             |
| `political-leaning`     | Left / Other / Right                                      |
| `conspiracy`            | Mentioned and promoted conspiracy theories                |
| `tropes`                | Detected narrative tropes                                 |
| `persuasion-techniques` | Detected persuasion techniques                            |
| `climate-related`       | Whether the text is climate-related                       |
| `frugal-ai-stance`      | Climate stance class                                      |

### Predict a Factor

```bash
curl -X POST "http://localhost:8000/predict/emotion" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Climate change is a serious threat to our planet."]}'
```

**Response**:

```json
{
  "results": [{ "value": "Fear" }],
  "processed_count": 1,
  "total_count": 1
}
```

`batch_size` and `max_length` can be passed in the payload to override the defaults for that request.

### Health Check

```bash
curl http://localhost:8000/health
```

### Model Information

```bash
curl http://localhost:8000/models
```

## Authentication

The API can be protected with a shared API key. Set `BERT_API_KEY` and every request must carry a matching `X-API-Key` header:

```bash
curl http://localhost:8000/health -H "X-API-Key: <your-key>"
```

When `BERT_API_KEY` is unset (or empty), the API is open.

## Configuration

Configure the server using environment variables (set in a `.env` file or directly in the environment):

| Variable                 | Description                                     | Default   |
| ------------------------ | ----------------------------------------------- | --------- |
| `BERT_MODELS_PATH`       | Path to model files                             | `models`  |
| `BERT_DEVICE`            | PyTorch device (auto/cpu/cuda)                  | `auto`    |
| `BERT_BATCH_SIZE`        | Default batch size                              | `32`      |
| `BERT_MAX_LENGTH`        | Default max sequence length                     | `128`     |
| `BERT_FRUGAL_BATCH_SIZE` | Batch size for `frugal-ai-stance`               | `8`       |
| `BERT_FRUGAL_MAX_LENGTH` | Max sequence length for `frugal-ai-stance`      | `256`     |
| `BERT_AUTO_DOWNLOAD`     | Auto-download missing models                    | `true`    |
| `BERT_API_KEY`           | Required `X-API-Key` value; empty disables auth | _(none)_  |
| `BERT_HOST`              | Server host                                     | `0.0.0.0` |
| `BERT_PORT`              | Server port                                     | `8000`    |
| `BERT_LOG_LEVEL`         | Logging level                                   | `INFO`    |

The server expects the following model checkpoints under `BERT_MODELS_PATH` (downloading automatically when `BERT_AUTO_DOWNLOAD=true`):

- `emotion.pth`
- `sentiment.pth`
- `political-leaning.pth`
- `conspiracy.pth`
- `tropes.pth`
- `persuasion-techniques.pth`

The `frugal-ai-stance` model is fetched directly from Hugging Face.

### Device Selection

To use a specific device, set the `BERT_DEVICE` variable like so:

```bash
BERT_DEVICE="cpu" # CPU-only
BERT_DEVICE="auto" # Automatically select GPUs if available
BERT_DEVICE="cuda:3"  # Use GPU #3
```
