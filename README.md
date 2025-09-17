# BERT Factors Server

A standalone HTTP API server for predicting emotion, sentiment, political leaning, and conspiracy factors using BERT models.

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
docker-compose up --build
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

### Predict Factors

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "texts": [
         "Climate change is a serious threat to our planet.",
         "I think the government is hiding the truth about vaccines."
       ],
       "batch_size": 32,
       "max_length": 128
     }'
```

**Response**:

```json
{
  "results": [
    {
      "emotion": "Fear",
      "sentiment": "Negative",
      "political_leaning": "Left",
      "conspiracies": {
        "mentioned": [],
        "promoted": []
      }
    },
    {
      "emotion": "Anger",
      "sentiment": "Negative",
      "political_leaning": "Right",
      "conspiracies": {
        "mentioned": ["Antivax"],
        "promoted": []
      }
    }
  ],
  "processed_count": 2,
  "total_count": 2
}
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Model Information

```bash
curl http://localhost:8000/models
```

## Configuration

Configure the server using environment variables:

| Variable             | Description                    | Default   |
| -------------------- | ------------------------------ | --------- |
| `BERT_MODELS_PATH`   | Path to model files            | `models`  |
| `BERT_DEVICE`        | PyTorch device (auto/cpu/cuda) | `auto`    |
| `BERT_BATCH_SIZE`    | Default batch size             | `32`      |
| `BERT_MAX_LENGTH`    | Default max sequence length    | `128`     |
| `BERT_AUTO_DOWNLOAD` | Auto-download missing models   | `true`    |
| `BERT_HOST`          | Server host                    | `0.0.0.0` |
| `BERT_PORT`          | Server port                    | `8000`    |
| `BERT_LOG_LEVEL`     | Logging level                  | `INFO`    |
