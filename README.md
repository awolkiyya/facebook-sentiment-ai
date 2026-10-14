# Municipal AI Service

AI inference service for the **Municipal Intelligence Platform**.

The Municipal AI Service is an independent Python/FastAPI service responsible for **AI model execution, inference, model lifecycle management, and AI-specific processing**.

It works together with Laravel, which remains the authoritative **application backend, business logic, orchestration, integration, workflow, and persistence layer**.

The service is designed initially for municipal social-media analysis and will progressively support:

* Language detection
* Sentiment analysis
* Topic classification
* Priority classification
* Complaint classification
* Summarization
* Trend analysis
* Other municipal AI capabilities

The architecture deliberately separates **application orchestration** from **AI inference** so that AI models can evolve independently without destabilizing the core Laravel application.

---

# 1. Architecture

The platform follows a strict separation between the application layer and the AI inference layer.

```text
┌─────────────────────────────────────────────────────────────┐
│                         Next.js                             │
│                      Web Dashboard                          │
│                                                             │
│  Analytics • Facebook • Sentiment • Topics • Reports       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               │ REST API
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                         Laravel                             │
│                  Application Backend                         │
│                                                             │
│  Authentication / Authorization                             │
│  Facebook / Meta Integration                                │
│  Data Collection                                            │
│  Application Business Logic                                │
│  Queue / Scheduler                                          │
│  AI Orchestration                                           │
│  Retry / Failure Handling                                   │
│  PostgreSQL Persistence                                     │
│                                                             │
│                     AIOrchestrator                          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               │ Internal HTTP API
                               │ X-Internal-API-Key
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Python AI Service                         │
│                         FastAPI                             │
│                                                             │
│  API Validation                                             │
│  Internal Service Security                                  │
│  Model Management                                           │
│  Device Management                                          │
│  Text Preprocessing                                         │
│  Tokenization                                               │
│  AI Inference                                               │
│  Probability / Confidence                                   │
│  Model Versioning                                           │
│  AI Observability                                           │
│                                                             │
│       ┌──────────────┬───────────────┬──────────────┐       │
│       │              │               │              │       │
│       ▼              ▼               ▼              │       │
│   Language       Sentiment          Topic            │       │
│    Model           Model            Model            │       │
└─────────────────────────────────────────────────────────────┘
```

---

# 2. Responsibility Boundary

## Laravel owns

Laravel is the **application and AI orchestration layer**.

It is responsible for:

* Authentication
* Authorization
* User and role management
* Facebook / Meta OAuth
* Facebook Page integration
* Facebook post collection
* Facebook comment collection
* Application business rules
* Queue processing
* Scheduling
* AI job orchestration
* AI request lifecycle
* Retry and failure handling
* Database persistence
* Auditability
* API consumed by Next.js

Laravel decides:

```text
WHAT should be analyzed

WHEN it should be analyzed

WHO is allowed to request it

WHERE the result should be stored

WHAT should happen when AI fails
```

Laravel should communicate with Python through an internal API rather than exposing AI model execution directly to frontend clients.

---

## Python owns

Python is the **AI inference layer**.

It is responsible for:

* Loading AI models
* Managing model lifecycle
* Selecting inference devices
* Model validation
* Text preprocessing required by models
* Tokenization
* Model inference
* Probability calculation
* Confidence calculation
* AI-specific result generation
* Model version identification
* AI-specific health checks
* AI observability

Python decides:

```text
HOW the AI prediction is calculated
```

Python does **not** own:

* Facebook integration
* Meta OAuth
* Application authentication
* Municipal business rules
* Application workflow
* Primary application database
* User permissions
* Laravel application authorization

---

# 3. AI Capability Architecture

The AI service is designed around independent model capabilities.

```text
Facebook Comment
       │
       ▼
Language Detection
       │
       ├── am → Amharic
       ├── om → Afaan Oromo
       └── en → English
       │
       ▼
Sentiment Analysis
       │
       ├── negative
       ├── neutral
       └── positive
       │
       ▼
Topic Classification
       │
       ├── water
       ├── roads
       ├── electricity
       ├── waste
       └── ...
       │
       ▼
Priority Classification
       │
       ▼
Complaint Classification
       │
       ▼
Structured Municipal Intelligence
```

Not every capability needs to be available at the same time.

Each capability can be:

* Developed independently
* Tested independently
* Versioned independently
* Loaded independently
* Deployed independently
* Replaced without redesigning the entire AI service

---

# 4. Current AI Capabilities

## Language Detection

The initial language detection capability is intended to identify:

```text
am → Amharic
om → Afaan Oromo
en → English
```

The initial implementation should use an existing language-identification model rather than immediately training a custom language classifier.

The current model direction is:

```text
cis-lmu/glotlid
```

The model should be evaluated against real municipal Facebook comments before being considered production-grade.

Particular attention should be given to:

* Short comments
* Spelling variations
* Social-media slang
* Code-switching
* Transliteration
* Afaan Oromo / Amharic ambiguity
* Mixed-language comments

A custom municipal language model should only be trained if evaluation demonstrates that the existing model is insufficient.

---

## Sentiment Analysis

Sentiment analysis is the first custom-trained AI capability.

The current fine-tuned model classifies text into:

```text
negative
neutral
positive
```

The current model was fine-tuned from:

```text
Davlan/afro-xlmr-base
```

The production service must load the **fine-tuned checkpoint**, not the original base model alone.

---

## Topic Classification

Topic classification will identify the municipal issue discussed by a citizen.

Potential municipal topics include:

```text
water
roads
electricity
waste_management
health
education
transport
housing
revenue
public_safety
government_service
other
```

These labels are examples of a proposed municipal taxonomy.

They must **not** be treated as the actual model labels until a topic dataset has been created, labeled, trained, and evaluated.

Unlike language detection, a useful municipal topic classifier will generally require a **municipal-specific labeled training dataset**.

---

# 5. Current Model Status

The service currently has one trained model:

```text
Sentiment Model
    ↓
afro-xlmr-sentiment
    ↓
v1.0.0
    ↓
Beta / Integration Testing
```

Language detection is an existing-model integration capability.

Topic classification is currently a development capability and requires a properly labeled municipal dataset before production inference.

Current status:

```text
Language Detection     → Integration / Evaluation
Sentiment              → Fine-tuned / Beta
Topic                   → Dataset / Training Required
Priority                → Planned
Complaint Classification → Planned
Summarization           → Planned
```

---

# 6. Current Sentiment Model

The initial sentiment model is based on:

```text
Davlan/afro-xlmr-base
```

The model was fine-tuned for a three-class sentiment classification task.

```text
Architecture:      XLM-R / Afro-XLM-R
Parameters:        ~278M
Task:              Sentiment Classification
Classes:           3
Framework:         PyTorch
Model ecosystem:  Hugging Face Transformers
```

The fine-tuned checkpoint contains the classification head required for sentiment prediction.

The service must therefore load the saved fine-tuned checkpoint produced by the training process.

---

# 7. Current Sentiment Evaluation

The current checkpoint is intended for:

* System testing
* Integration testing
* Pipeline validation
* API testing
* Facebook ingestion testing
* End-to-end workflow testing

Current validation results:

```text
Accuracy:       49.97%
Macro F1:       49.50%
Weighted F1:    56.62%
```

Per-class validation performance:

```text
              Precision    Recall    F1       Support

negative      0.9507       0.4473   0.6083    1337

neutral       0.1835       0.9241   0.3062     224

positive      0.8017       0.4429   0.5706     438
```

The results indicate significant class imbalance and uneven class performance.

In particular, the neutral class has high recall but low precision.

Therefore:

> **Sentiment v1.0.0 is a beta/test model and must not currently be treated as a production-certified decision-making model.**

The immediate objective is to validate the complete technical pipeline.

Further dataset development, balancing, training, and evaluation are required before high-confidence municipal decision support.

---

# 8. Model Versioning

Every AI model is independently versioned.

Recommended structure:

```text
models/
├── language/
│   └── v1.0.0/
│
├── sentiment/
│   └── v1.0.0/
│
└── topic/
    └── v1.0.0/
```

A model version directory contains its complete model artifact.

Example:

```text
models/
└── sentiment/
    └── v1.0.0/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── special_tokens_map.json
        └── manifest.json
```

A new model must receive a new version rather than silently replacing an existing artifact.

For example:

```text
v1.0.0
v1.1.0
v2.0.0
```

Historical model versions should remain available when practical so that previous analyses remain reproducible.

---

# 9. Model Manifest

Every versioned model should contain a `manifest.json`.

The manifest describes the model artifact independently of application code.

Example:

```json
{
  "name": "afro-xlmr-sentiment",
  "version": "1.0.0",
  "display_name": "Afro-XLMR Sentiment Classifier",

  "task": {
    "type": "text-classification",
    "subtask": "sentiment-analysis",
    "num_labels": 3
  },

  "base_model": {
    "name": "Davlan/afro-xlmr-base",
    "architecture": "XLM-RoBERTa"
  },

  "labels": {
    "0": "negative",
    "1": "neutral",
    "2": "positive"
  },

  "languages": [
    "am"
  ],

  "inference": {
    "max_length": 128,
    "truncation": true,
    "probability_function": "softmax",
    "prediction_strategy": "argmax"
  },

  "status": "beta",

  "runtime": {
    "supported_devices": [
      "cuda",
      "mps",
      "cpu"
    ],
    "default_device": "auto"
  }
}
```

The manifest allows the service to validate:

* Model identity
* Version
* Task
* Number of labels
* Label mapping
* Supported languages
* Inference configuration
* Runtime requirements
* Model status

The model configuration itself remains authoritative for the actual label mapping.

---

# 10. Independent Model Versioning

Language, sentiment, and topic models must not share one global version.

For example:

```text
Language:
    municipal-language-classifier v1.0.0

Sentiment:
    afro-xlmr-sentiment v1.0.0

Topic:
    municipal-topic-classifier v2.0.0
```

A future deployment might therefore use:

```text
Language   → v1.1.0
Sentiment  → v1.2.0
Topic      → v2.0.0
```

This allows each model to evolve independently.

---

# 11. Model Traceability

Every AI prediction should identify the model that generated it.

Example:

```json
{
  "model": {
    "name": "afro-xlmr-sentiment",
    "version": "1.0.0"
  }
}
```

For a complete analysis, Laravel should store the model identity for each capability.

For example:

```text
language_model
language_model_version

sentiment_model
sentiment_model_version

topic_model
topic_model_version
```

This provides historical traceability.

Example:

```text
Facebook Comment
        ↓
Analysis created on 2026-09-07
        ↓
Language Model:  v1.0.0
Sentiment Model: v1.0.0
Topic Model:     v1.0.0
```

If a model is retrained later, historical analyses remain associated with the model version that actually produced them.

---

# 12. Project Structure

```text
facebook-sentiment-ai/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── health.py
│   │       └── analysis.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── exceptions.py
│   │   └── security.py
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── analysis.py
│   │   └── health.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── analysis_service.py
│   │   ├── language_service.py
│   │   ├── sentiment_service.py
│   │   ├── topic_service.py
│   │   └── priority_service.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── model_manager.py
│   │   │
│   │   ├── language/
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   ├── predictor.py
│   │   │   └── config.py
│   │   │
│   │   ├── sentiment/
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   ├── predictor.py
│   │   │   └── config.py
│   │   │
│   │   └── topic/
│   │       ├── __init__.py
│   │       ├── model.py
│   │       ├── predictor.py
│   │       └── config.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── text_cleaner.py
│   │   ├── normalizer.py
│   │   └── tokenizer.py
│   │
│   ├── postprocessing/
│   │   ├── __init__.py
│   │   ├── sentiment_mapper.py
│   │   └── confidence.py
│   │
│   └── observability/
│       ├── __init__.py
│       ├── metrics.py
│       └── tracing.py
│
├── models/
│   ├── language/
│   │   └── v1.0.0/
│   │
│   ├── sentiment/
│   │   └── v1.0.0/
│   │
│   └── topic/
│       └── v1.0.0/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── scripts/
│   ├── download_models.py
│   └── verify_models.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── .dockerignore
└── README.md
```

The structure intentionally separates:

```text
API
  ↓
Application Services
  ↓
Inference Layer
  ↓
Models
```

---

# 13. API Versioning

All application endpoints use explicit API versioning.

Current namespace:

```text
/api/v1
```

Future versions may coexist:

```text
/api/v1
/api/v2
```

This allows the API contract to evolve without unnecessarily breaking Laravel integrations.

---

# 14. Health API

## Liveness

```http
GET /api/v1/health/live
```

Response:

```json
{
  "status": "ok"
}
```

Liveness determines whether the Python process is running.

It should remain lightweight and should not require successful model inference.

---

## Readiness

```http
GET /api/v1/health/ready
```

Readiness should eventually verify:

```text
Application running
        +
Required models loaded
        +
Model validation successful
        +
Required resources available
        =
READY
```

A process that is alive but cannot perform required inference must not report itself as ready.

---

# 15. AI Analysis API

The primary inference endpoint is:

```http
POST /api/v1/analyze
```

Example request:

```json
{
  "text": "Bishaan nuuf hin dhufu, rakkoon kun maalif hin furamne?"
}
```

The response should contain only capabilities that are actually available.

For the initial sentiment-only implementation:

```json
{
  "sentiment": {
    "label": "negative",
    "confidence": 0.91,
    "probabilities": {
      "negative": 0.91,
      "neutral": 0.04,
      "positive": 0.05
    }
  },
  "model": {
    "name": "afro-xlmr-sentiment",
    "version": "1.0.0"
  }
}
```

When language detection becomes available:

```json
{
  "language": {
    "code": "om",
    "confidence": 0.98
  },
  "sentiment": {
    "label": "negative",
    "confidence": 0.91
  },
  "models": {
    "language": {
      "name": "municipal-language-classifier",
      "version": "1.0.0"
    },
    "sentiment": {
      "name": "afro-xlmr-sentiment",
      "version": "1.0.0"
    }
  }
}
```

Topic and priority fields must not be fabricated before their models are implemented.

---

# 16. Inference Lifecycle

Models must **never be loaded for every request**.

The intended lifecycle is:

```text
Application Startup
        ↓
ModelManager
        ↓
Discover Model Configuration
        ↓
Load Model Manifest
        ↓
Load Tokenizer
        ↓
Load Model
        ↓
Select Device
        ↓
Move Model to Device
        ↓
Validate Model
        ↓
Application Ready
        ↓
Receive Request
        ↓
Preprocess Text
        ↓
Run Inference
        ↓
Calculate Probabilities
        ↓
Calculate Confidence
        ↓
Return Structured Result
```

This prevents expensive model initialization for every Facebook comment.

---

# 17. ModelManager

`ModelManager` owns the lifecycle of AI models.

Responsibilities include:

```text
Discover model
Load manifest
Load tokenizer
Load model
Select device
Validate model
Validate label mapping
Expose metadata
Keep model in memory
Provide model to predictors
```

Conceptually:

```text
ModelManager
│
├── Language Model
│
├── Sentiment Model
│
├── Topic Model
│
└── Future Models
```

A predictor should request an already-loaded model from the `ModelManager`.

The predictor should not independently reload the model for every request.

---

# 18. Device Management

The inference layer supports:

```text
CPU
CUDA
Apple MPS
```

The preferred configuration is:

```env
MODEL_DEVICE=auto
```

Automatic selection should follow an environment-appropriate strategy:

```text
NVIDIA GPU available
        ↓
CUDA

Apple Silicon GPU available
        ↓
MPS

Otherwise
        ↓
CPU
```

Model-specific code should not hard-code:

```python
cuda
```

or:

```python
cpu
```

The selected device should be controlled through configuration and hardware detection.

---

# 19. Text Processing Pipeline

The general processing pipeline is:

```text
Raw Text
   ↓
Input Validation
   ↓
Text Cleaning
   ↓
Normalization
   ↓
Language Detection
   ↓
Capability-specific Preprocessing
   ↓
Tokenization
   ↓
Model Inference
   ↓
Logits
   ↓
Softmax Probability
   ↓
Label Mapping
   ↓
Confidence
   ↓
Structured AI Result
```

The original Facebook comment remains the responsibility of Laravel and should remain available there for auditing and traceability.

The Python service should process the text necessary for inference and return structured AI results.

---

# 20. Sentiment Prediction

The current sentiment inference logic follows the Transformer sequence-classification pattern:

```python
model.eval()

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=128,
)

with torch.no_grad():
    outputs = model(**inputs)

probabilities = F.softmax(
    outputs.logits,
    dim=-1,
)

predicted_id = torch.argmax(
    probabilities,
    dim=-1,
)
```

The resulting prediction contains:

```text
Sentiment label
Confidence
Probability for each class
Model name
Model version
```

Example:

```json
{
  "sentiment": {
    "label": "negative",
    "confidence": 0.91,
    "probabilities": {
      "negative": 0.91,
      "neutral": 0.04,
      "positive": 0.05
    }
  }
}
```

The label mapping must be obtained from the actual trained model configuration and validated against the model manifest.

---

# 21. Language Detection

Language detection is a separate AI capability.

Its responsibility is:

```text
Input Text
    ↓
Language Identification
    ↓
Language Code
    ↓
Confidence
```

Target languages:

```text
am → Amharic
om → Afaan Oromo
en → English
```

The initial implementation should use an existing text language-identification model such as:

```text
cis-lmu/glotlid
```

The service should not assume that a general language-identification model is automatically production-accurate for municipal Facebook content.

It must be evaluated using representative municipal data.

Evaluation should include:

* Standard text
* Short comments
* Informal spelling
* Code-switching
* Transliteration
* Social-media expressions
* Afaan Oromo
* Amharic
* English

A custom language model can be introduced later if required.

---

# 22. Topic Classification

Topic classification is a separate model from sentiment.

Its responsibility is:

```text
Citizen Comment
      ↓
Topic Model
      ↓
Municipal Issue
```

Potential topics include:

```text
water
roads
electricity
waste_management
health
education
transport
housing
revenue
public_safety
government_service
other
```

These labels represent the planned municipal taxonomy and must be finalized using real municipal requirements and labeled data.

A topic model generally requires:

```text
Real municipal comments
        ↓
Human labeling
        ↓
Dataset validation
        ↓
Train / validation / test split
        ↓
Model fine-tuning
        ↓
Evaluation
        ↓
Model version
        ↓
Deployment
```

Topic classification should therefore be treated as a separate ML development track rather than assuming the sentiment model can perform topic classification.

---

# 23. Confidence Handling

Every prediction should expose model confidence where supported.

Example:

```json
{
  "label": "negative",
  "confidence": 0.91
}
```

Confidence is derived from the model's output probabilities.

Future versions may categorize predictions:

```text
High Confidence
        ↓
Acceptable Confidence
        ↓
Uncertain
        ↓
Human Review
```

Low-confidence results may eventually be routed to human review.

Confidence must not be interpreted as guaranteed correctness.

---

# 24. Internal Security

The Python AI service is an internal service.

Preferred architecture:

```text
Internet
   ↓
Next.js
   ↓
Laravel
   ↓
Private / Internal Network
   ↓
Python AI Service
```

The Python service should not be publicly exposed unless explicitly required.

## Internal API Authentication

Laravel authenticates to Python using an internal API key:

```http
X-Internal-API-Key: <secret>
```

Current security behavior:

```text
Missing key
    ↓
401 Unauthorized

Invalid key
    ↓
403 Forbidden

Valid key
    ↓
Request accepted
```

The secret must:

* Never be hard-coded
* Never be committed to Git
* Never be placed in frontend code
* Never be exposed to browsers
* Be rotated when compromised
* Be different between environments where appropriate

Example:

```env
INTERNAL_API_KEY=<long-random-secret>
```

---

# 25. Configuration

Configuration is environment-driven.

Example:

```env
# =============================================================================
# Municipal AI Service
# =============================================================================

APP_NAME=Municipal AI Service
APP_VERSION=1.0.0
APP_ENV=local
APP_DEBUG=false

API_PREFIX=/api/v1

HOST=0.0.0.0
PORT=8000

# =============================================================================
# AI Models
# =============================================================================

MODEL_DEVICE=auto

LANGUAGE_MODEL_NAME=cis-lmu/glotlid
LANGUAGE_MODEL_VERSION=1.0.0

SENTIMENT_MODEL_NAME=afro-xlmr-sentiment
SENTIMENT_MODEL_VERSION=1.0.0

TOPIC_MODEL_NAME=municipal-topic-classifier
TOPIC_MODEL_VERSION=1.0.0

MODEL_DIR=./models

# =============================================================================
# Internal Security
# =============================================================================

INTERNAL_API_KEY=<secret>

# =============================================================================
# Inference
# =============================================================================

MAX_TEXT_LENGTH=5000
INFERENCE_TIMEOUT_SECONDS=30

# =============================================================================
# Logging
# =============================================================================

LOG_LEVEL=INFO
```

The Topic model configuration may exist before the model is implemented, but the service must not report Topic inference as available until its model artifact has been deployed and validated.

Production secrets must never be committed to source control.

The `.env` file must be excluded through `.gitignore`.

---

# 26. Structured Logging

The service uses `structlog` for structured application logging.

Important fields include:

```text
request_id
endpoint
status
model
model_version
language
latency
error
```

Example conceptual log:

```json
{
  "event": "inference_completed",
  "request_id": "abc123",
  "model": "afro-xlmr-sentiment",
  "model_version": "1.0.0",
  "latency_ms": 184,
  "status": "success"
}
```

Structured logging allows Docker, log aggregation systems, and observability platforms to process logs reliably.

---

# 27. Request Tracing

The target system should propagate a request ID across the complete operation.

```text
Next.js
   ↓
Laravel
   ↓
Queue
   ↓
AIOrchestrator
   ↓
Python AI Service
   ↓
ModelManager
   ↓
Inference
```

The preferred header is:

```http
X-Request-ID: <uuid>
```

Python should:

1. Accept an existing request ID.
2. Generate one when absent.
3. Bind it to structured logging context.
4. Include it in relevant logs.
5. Return it when appropriate.

This allows a single Facebook comment analysis to be traced across the entire platform.

---

# 28. Error Handling

The service should use explicit application exceptions.

Examples:

```text
InvalidRequest
AuthenticationError
ModelNotLoaded
ModelLoadError
ModelValidationError
ModelInferenceError
UnsupportedLanguage
InferenceTimeout
ServiceUnavailable
```

Internal stack traces must not be exposed directly to API clients.

API errors should be predictable and machine-readable.

Example:

```json
{
  "error": {
    "code": "MODEL_NOT_LOADED",
    "message": "Sentiment model is not available."
  }
}
```

Detailed stack traces remain in server-side logs.

The service must never silently return fabricated AI results when inference fails.

---

# 29. Testing Strategy

Testing is divided into three levels.

## Unit Tests

Individual components should be tested independently:

```text
TextCleaner

Normalizer

ConfidenceCalculator

LanguagePredictor

SentimentPredictor

TopicPredictor

ResultMapper

ModelManager

Security
```

## Integration Tests

Test the interaction between:

```text
FastAPI
   ↓
AnalysisService
   ↓
Inference Layer
   ↓
Model
```

## API Tests

Test:

```text
GET  /api/v1/health/live

GET  /api/v1/health/ready

POST /api/v1/analyze
```

Security tests should include:

```text
No API key
Wrong API key
Correct API key
```

Language fixtures should include:

```text
Afaan Oromo
Amharic
English
```

Sentiment fixtures should cover:

```text
Negative
Neutral
Positive
```

Topic fixtures should be introduced after the topic dataset has been established.

---

# 30. Docker

The AI service is designed to run independently in Docker.

Production deployment should provide:

```text
Python Runtime
      ↓
FastAPI
      ↓
PyTorch
      ↓
Transformers
      ↓
Model Artifacts
      ↓
Environment Configuration
      ↓
Health Checks
      ↓
Structured Logging
```

GPU deployments may use NVIDIA CUDA-compatible infrastructure.

Model artifacts should be managed separately from application source code where appropriate.

Large model files should not be unnecessarily committed to Git repositories.

---

# 31. Development Environment

Create the virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the development server:

```bash
uvicorn app.main:app \
    --reload \
    --host 0.0.0.0 \
    --port 8000
```

API documentation:

```text
http://localhost:8000/docs
```

Liveness endpoint:

```text
http://localhost:8000/api/v1/health/live
```

Readiness endpoint:

```text
http://localhost:8000/api/v1/health/ready
```

---

# 32. Model Installation

The fine-tuned sentiment model should be extracted into its versioned directory.

Target structure:

```text
models/
└── sentiment/
    └── v1.0.0/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── special_tokens_map.json
        └── manifest.json
```

The model directory must contain the **fine-tuned checkpoint**, including the trained classification head.

The base model:

```text
Davlan/afro-xlmr-base
```

should not be treated as equivalent to the fine-tuned sentiment checkpoint.

---

# 33. Current Development Status

```text
[x] Project structure

[x] FastAPI application

[x] Configuration system

[x] Liveness endpoint

[x] Readiness endpoint foundation

[x] Structured logging

[x] Internal API security

[x] Environment-based secrets

[x] Python virtual environment

[x] PyTorch installation

[x] Transformers installation

[x] Apple MPS detection

[x] Sentiment model training

[x] Sentiment model evaluation

[ ] ModelManager

[ ] Model manifest loading

[ ] Model validation

[ ] DeviceManager

[ ] Fine-tuned sentiment model loading

[ ] Language model integration

[ ] Language model evaluation

[ ] Sentiment inference API

[ ] AnalysisService

[ ] Confidence handling

[ ] Request ID tracing

[ ] Topic dataset

[ ] Topic model training

[ ] Topic inference

[ ] Unit tests

[ ] Integration tests

[ ] Docker

[ ] GPU deployment

[ ] Laravel integration
```

The fine-tuned sentiment model is currently suitable for **initial integration testing**.

---

# 34. Implementation Roadmap

## Phase 1 — Service Foundation

```text
FastAPI
   ↓
Configuration
   ↓
Health
   ↓
Structured Logging
   ↓
Exception Handling
   ↓
Internal API Security
```

**Status: Foundation implemented**

---

## Phase 2 — Model Infrastructure

```text
ModelManager
      ↓
Model Manifest
      ↓
Device Detection
      ↓
Model Loading
      ↓
Tokenizer Loading
      ↓
Model Validation
      ↓
Version Identification
```

**Next major implementation phase**

---

## Phase 3 — Sentiment Inference

```text
Input Text
   ↓
Preprocessing
   ↓
Tokenizer
   ↓
Fine-tuned Afro-XLM-R
   ↓
Logits
   ↓
Softmax
   ↓
Sentiment Label
   ↓
Confidence
```

---

## Phase 4 — Language Detection

```text
Input Text
   ↓
Language Model
   ↓
Language Code
   ↓
Confidence
```

Initial target:

```text
am
om
en
```

The selected language model must be evaluated against representative municipal Facebook content.

---

## Phase 5 — Analysis API

```text
POST /api/v1/analyze
        ↓
Authentication
        ↓
Request Validation
        ↓
AnalysisService
        ↓
Language Predictor
        ↓
Sentiment Predictor
        ↓
Structured Result
```

Only available and validated capabilities should be included in the response.

---

## Phase 6 — Laravel Integration

```text
Facebook Comment
        ↓
Laravel Queue
        ↓
AIOrchestrator
        ↓
Python AI Service
        ↓
AI Result
        ↓
Laravel
        ↓
PostgreSQL
```

Laravel remains responsible for storing and managing the result.

---

## Phase 7 — Topic Intelligence

```text
Real Facebook Comments
        ↓
Human Annotation
        ↓
Municipal Topic Dataset
        ↓
Training
        ↓
Evaluation
        ↓
Topic Model v1.0.0
        ↓
Python AI Service
```

Topic classification should be based on real municipal requirements rather than arbitrary labels.

---

## Phase 8 — Additional AI Capabilities

```text
Language Detection
        ↓
Sentiment Analysis
        ↓
Topic Classification
        ↓
Priority Classification
        ↓
Complaint Classification
        ↓
Summarization
        ↓
Trend Analysis
```

Each capability should be independently versioned.

---

## Phase 9 — Production Hardening

```text
Docker
   ↓
GPU Support
   ↓
Request IDs
   ↓
Metrics
   ↓
Tracing
   ↓
Timeouts
   ↓
Retry Strategy
   ↓
Resource Limits
   ↓
Model Monitoring
   ↓
Security Hardening
```

---

# 35. Design Principles

## Separation of Concerns

```text
Laravel
    =
Application + Business Logic + Orchestration

Python
    =
AI Inference + Model Lifecycle
```

---

## Independent Model Lifecycle

Each AI capability has its own model lifecycle.

```text
Language
    ↓
Version independently

Sentiment
    ↓
Version independently

Topic
    ↓
Version independently
```

A new sentiment model does not require a new language model version.

---

## Model Reuse

Models are loaded once and reused.

Bad:

```text
Request
  ↓
Load Model
  ↓
Predict
  ↓
Unload Model
```

Good:

```text
Application Startup
  ↓
Load Model
  ↓
Keep Model in Memory
  ↓
Request → Predict
  ↓
Request → Predict
  ↓
Request → Predict
```

---

## Versioned Models

Every AI prediction must identify the model version that generated it.

```json
{
  "model": {
    "name": "afro-xlmr-sentiment",
    "version": "1.0.0"
  }
}
```

---

## Model Manifest

Every versioned model should contain metadata describing:

```text
Model name
Model version
Task
Labels
Languages
Inference configuration
Runtime
Status
```

This improves validation, traceability, and reproducibility.

---

## Configuration over Hard-Coding

Deployment-specific behavior belongs in environment configuration.

Examples:

```text
Model device

Model path

Model version

API keys

Logging level

API prefix

Maximum text length

Inference timeout
```

---

## Fail Safely

Model failures must result in controlled errors.

The service must never silently return fabricated predictions.

---

## Observable by Default

Inference operations should be traceable through:

```text
Request ID

Model

Model Version

Latency

Status

Errors
```

---

## Production First

Development shortcuts should not become permanent architecture.

The current beta sentiment model can be used to validate the technical pipeline, but model quality must be validated before the system is used for high-impact operational decisions.

---

# 36. Target Production Flow

The complete platform will eventually operate as:

```text
                         Facebook / Meta
                               │
                               ▼
                         Laravel Backend
                               │
                         Facebook Data
                               │
                               ▼
                            Queue
                               │
                               ▼
                        AIOrchestrator
                               │
                     Internal HTTP Request
                               │
                    X-Internal-API-Key
                               │
                               ▼
                    Python AI Service
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
                 ▼             ▼             ▼
             Language      Sentiment       Topic
               Model          Model         Model
                 │             │             │
                 └─────────────┼─────────────┘
                               │
                               ▼
                      Priority Classification
                               │
                               ▼
                    Complaint Classification
                               │
                               ▼
                       Structured AI Result
                               │
                               ▼
                           Laravel
                               │
                               ▼
                         PostgreSQL
                               │
                               ▼
                           Next.js
                               │
                               ▼
                    Municipal Intelligence
                           Dashboard
```

---

# 37. Architectural Contract

The most important architectural contract is:

```text
┌────────────────────────────────────────────┐
│                 Laravel                    │
│                                            │
│  WHAT                                      │
│  WHEN                                      │
│  WHO                                       │
│  WHERE                                     │
│  WORKFLOW                                  │
│  BUSINESS RULES                            │
│  PERSISTENCE                               │
│  ORCHESTRATION                              │
└──────────────────────┬─────────────────────┘
                       │
                       │ Internal AI API
                       ▼
┌────────────────────────────────────────────┐
│              Python AI Service             │
│                                            │
│  HOW                                       │
│  MODEL                                     │
│  TOKENIZATION                              │
│  PREPROCESSING                             │
│  INFERENCE                                 │
│  PROBABILITY                               │
│  CONFIDENCE                                │
│  MODEL VERSION                             │
│  AI HEALTH                                 │
└────────────────────────────────────────────┘
```

Laravel remains the **application and AI orchestration layer**.

Python remains the **AI inference and model lifecycle layer**.

---

# 38. Final Objective

The ultimate objective is to provide Adama City's municipal administration with a centralized intelligence platform capable of transforming large volumes of public digital feedback into structured, actionable information.

The long-term platform should enable:

```text
Public Feedback
       ↓
Data Collection
       ↓
Language Detection
       ↓
Sentiment Analysis
       ↓
Topic Classification
       ↓
Priority Classification
       ↓
Complaint Classification
       ↓
Trend Detection
       ↓
Semantic Analysis
       ↓
Municipal Intelligence
       ↓
Decision Support
```

The AI service is therefore designed as a **dedicated, versioned inference platform**, while Laravel remains the authoritative application, orchestration, workflow, and persistence layer.

This architecture allows new AI capabilities and improved models to be introduced without redesigning the core municipal application.

Municipal AI Service
MVP Version: 1.0.0

Language:  am / om / en
Sentiment: negative / neutral / positive
Topics:    25 municipal topics

Status:
✅ API operational
✅ Models loaded
✅ End-to-end inference
✅ Multilingual analysis
✅ Sentiment analysis
✅ Topic classification
✅ Batch analysis
✅ Model metadata


STEP 1
Ollama installed and GPU verified
        ↓
STEP 2
Benchmark 2–3 local models
        ↓
STEP 3
Create InsightProvider interface
        ↓
STEP 4
Create structured prompt
        ↓
STEP 5
Post summary endpoint
        ↓
STEP 6
Generate structured JSON
        ↓
STEP 7
Store generated insight
        ↓
STEP 8
Display insight on dashboard
        ↓
STEP 9
Evaluate hallucination/quality
        ↓
STEP 10
7-day / 30-day municipal intelligence


You are a municipal intelligence assistant.

Return ONLY valid JSON. No markdown.

Analyze this information:

Post:
"The municipality has started repairing the road in our neighborhood."

Comments:
1. "Thank you for repairing our road."
2. "Please finish the work quickly because the road is still difficult to use."

Return exactly this structure:

{
  "summary": "...",
  "key_concerns": [],
  "positive_signals": [],
  "public_requests": [],
  "attention_areas": []
}

Use only information explicitly provided. Do not invent facts.


llama3.2:3b
qwen2.5:7b