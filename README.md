# 🎧 AudioRAG: Audio-to-Insight AI Agent

![AudioRAG Screenshot 1](https://github.com/user-attachments/assets/c24b612f-984b-4d48-84c0-10ed5d19207b)
![AudioRAG Screenshot 2](https://github.com/user-attachments/assets/407c2d36-4a0d-4147-ac5b-db6e71efd7bb)
![AudioRAG Screenshot 3](https://github.com/user-attachments/assets/8aa5d0ab-ea91-427f-aa41-7e765cfc336a)
![AI as Service Report](https://github.com/user-attachments/assets/d98080d4-d5cb-432d-994e-2e5f8e15529d)

---

## 🔍 Features

### Audio Ingestion  
- Supports MP3, WAV, and M4A formats  
- Preprocessing: resampling, normalization, mono conversion  

### Transcription & Speaker Diarization  
- Powered by **AssemblyAI**  
- Accurate transcription with multi-speaker identification  

### RAG Framework  
- Natural language querying of audio content using:  
  - **Qdrant** for vector-based semantic search  
  - **SambaNova GPT-3.5** for context-aware generative responses  

### Text Chunking & Embedding  
- Transcripts segmented into semantic chunks  
- Embeddings generated via **OpenAI** / **HuggingFace** models  
- Stored in **Qdrant** for efficient retrieval  

### User Interface  
- Built with **Streamlit**  
- Features:  
  - Audio upload  
  - Transcript viewing  
  - Query interface  
  - Debugging tools  

### Manual Overrides  
- Customize diarization by specifying speaker counts  
- Review and correct diarization outputs  

### Timestamp Filtering  
- Time-based search capability  
- Session logs for improved analysis  

---

## 📌 Applications

- 🎙️ **Journalism & Media**  
  Transcribe interviews and podcasts with speaker attribution  

- ⚖️ **Legal & Compliance**  
  Convert courtroom or deposition recordings into searchable transcripts  

- 📞 **Customer Support**  
  Analyze call interactions for quality assurance  

- 🏢 **Corporate Meetings**  
  Summarize discussions and track action items  

- 🎓 **Research & Academia**  
  Query and analyze lectures or interviews with ease  

---

## 🛠️ Tech Stack

| Component        | Technology Used          |
|------------------|-------------------------|
| Programming      | Python                  |
| Transcription    | AssemblyAI              |
| Semantic Search  | Qdrant                  |
| Embeddings       | OpenAI / HuggingFace    |
| LLM Responses    | SambaNova GPT-3.5       |
| User Interface   | Streamlit               |
| Audio Processing | Librosa / Pydub         |

---

