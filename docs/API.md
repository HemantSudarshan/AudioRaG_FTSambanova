# AudioRAG API Documentation

## Overview

AudioRAG provides a REST API for programmatic access to audio analytics capabilities.

**Base URL:** `http://localhost:8000/api/v1`

---

## Authentication

All API requests require an API key in the header:

```http
Authorization: Bearer <api_key>
```

---

## Endpoints

### Audio Management

#### Upload Audio
```http
POST /audio
Content-Type: multipart/form-data
```

**Request:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | File | Yes | Audio file (MP3, WAV, M4A) |
| name | string | No | Custom name for the audio |
| speakers | integer | No | Expected number of speakers |

**Response:**
```json
{
  "id": "audio_123abc",
  "name": "interview.mp3",
  "status": "processing",
  "created_at": "2026-01-05T10:00:00Z"
}
```

---

#### Get Audio Details
```http
GET /audio/{audio_id}
```

**Response:**
```json
{
  "id": "audio_123abc",
  "name": "interview.mp3",
  "status": "completed",
  "duration_seconds": 1800,
  "speakers_detected": 2,
  "transcript_segments": 145,
  "created_at": "2026-01-05T10:00:00Z"
}
```

---

#### Get Transcript
```http
GET /audio/{audio_id}/transcript
```

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| format | string | json | Output format: json, txt, srt |
| speaker | string | all | Filter by speaker |

**Response (JSON):**
```json
{
  "segments": [
    {
      "speaker": "Speaker A",
      "text": "Hello, welcome to the interview.",
      "start_time": 0.0,
      "end_time": 3.5,
      "confidence": 0.95
    }
  ]
}
```

---

### Query API

#### Query Audio Content
```http
POST /query
```

**Request:**
```json
{
  "audio_id": "audio_123abc",
  "query": "What was discussed about the project timeline?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "The project timeline was discussed in detail...",
  "sources": [
    {
      "text": "[02:30] Speaker A: The timeline for the project...",
      "score": 0.89
    }
  ]
}
```

---

### Batch Processing

#### Submit Batch Job
```http
POST /batch
```

**Request:**
```json
{
  "audio_ids": ["audio_1", "audio_2", "audio_3"],
  "operations": ["transcribe", "summarize"]
}
```

**Response:**
```json
{
  "batch_id": "batch_456def",
  "status": "queued",
  "total_files": 3
}
```

---

#### Get Batch Status
```http
GET /batch/{batch_id}
```

**Response:**
```json
{
  "batch_id": "batch_456def",
  "status": "processing",
  "progress": 66,
  "completed": 2,
  "total": 3
}
```

---

### Analytics

#### Get Usage Analytics
```http
GET /analytics
```

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| start_date | date | 30 days ago | Start of date range |
| end_date | date | today | End of date range |

**Response:**
```json
{
  "total_audio_hours": 45.5,
  "total_queries": 1234,
  "average_response_time_ms": 850,
  "active_users": 12
}
```

---

## Rate Limits

| Tier | Requests/Hour | Concurrent Uploads |
|------|---------------|-------------------|
| Free | 100 | 1 |
| Pro | 1,000 | 5 |
| Enterprise | Custom | Custom |

Rate limit headers included in responses:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704456000
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid API key |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource doesn't exist |
| 429 | Too Many Requests - Rate limited |
| 500 | Internal Server Error |

**Error Response Format:**
```json
{
  "error": {
    "code": "INVALID_AUDIO_FORMAT",
    "message": "Uploaded file must be MP3, WAV, or M4A",
    "details": {
      "received_format": "ogg"
    }
  }
}
```

---

## Webhooks

Configure webhook endpoints to receive notifications:

```http
POST /webhooks
```

**Request:**
```json
{
  "url": "https://your-server.com/webhook",
  "events": ["audio.completed", "batch.completed"]
}
```

**Webhook Payload:**
```json
{
  "event": "audio.completed",
  "timestamp": "2026-01-05T10:05:00Z",
  "data": {
    "audio_id": "audio_123abc",
    "status": "completed"
  }
}
```
