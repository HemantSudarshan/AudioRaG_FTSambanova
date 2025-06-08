# 🎧 AudioRAG: Audio-to-Insight

---

![AudioRAG Screenshot 1](https://github.com/user-attachments/assets/c24b612f-984b-4d48-84c0-10ed5d19207b)  
![AudioRAG Screenshot 2](https://github.com/user-attachments/assets/407c2d36-4a0d-4147-ac5b-db6e71efd7bb)  
![AudioRAG Screenshot 3](https://github.com/user-attachments/assets/8aa5d0ab-ea91-427f-aa41-7e765cfc336a)  
![AI as Service Report](https://github.com/user-attachments/assets/d98080d4-d5cb-432d-994e-2e5f8e15529d)

---

## 📖 Overview

**AudioRAG** is a cutting-edge AI-powered tool that converts audio content into actionable insights by combining transcription, speaker diarization, and natural language querying. This project, developed by **Hemant Sudarshan**  harnesses Retrieval-Augmented Generation (RAG) and SambaNova Systems’ advanced AI infrastructure for seamless audio analytics.

---

## 🔍 Features

### 🎙️ Audio Ingestion
- Supports **MP3**, **WAV**, and **M4A** formats  
- Preprocessing includes **resampling**, **normalization**, and **mono conversion**

### 📝 Transcription & Speaker Diarization
- Powered by **AssemblyAI**  
- Highly accurate transcription with multi-speaker identification

### 🔄 RAG Framework
- Natural language querying over audio content with:  
  - **Qdrant** for vector-based semantic search  
  - **SambaNova GPT-3.5** for context-aware generative responses

### 🧩 Text Chunking & Embedding
- Transcripts segmented into semantic chunks  
- Embeddings created with **OpenAI** / **HuggingFace** models  
- Stored efficiently in **Qdrant**

### 🖥️ User Interface
- Built with **Streamlit**  
- Features intuitive:  
  - Audio upload  
  - Transcript viewing  
  - Query interface  
  - Debugging tools

### ✍️ Manual Overrides
- Customize diarization by specifying speaker counts  
- Review and correct diarization outputs for accuracy

### ⏰ Timestamp Filtering
- Enables time-based search functionality  
- Generates session logs for streamlined analysis

---

## 📌 Applications

| Use Case             | Description                                      |
|----------------------|------------------------------------------------|
| 🎙️ Journalism & Media   | Transcribe interviews and podcasts with speaker attribution |
| ⚖️ Legal & Compliance   | Convert courtroom or deposition recordings into searchable transcripts |
| 📞 Customer Support     | Analyze call interactions for insights and quality assurance |
| 🏢 Corporate Meetings   | Summarize discussions and track follow-up tasks |
| 🎓 Research & Academia  | Query and analyze lectures or interviews easily |

---

## 🛠️ Tech Stack

| Component         | Technology Used         |
|-------------------|------------------------|
| Programming       | Python                 |
| Transcription     | AssemblyAI             |
| Semantic Search   | Qdrant                 |
| Embeddings        | OpenAI / HuggingFace   |
| LLM Responses     | SambaNova GPT-3.5      |
| User Interface    | Streamlit              |
| Audio Processing  | Librosa / Pydub        |

---

Download the embedding model from https://www.mediafire.com/folder/41tpj4qu5ltyd/hf_cache, extract it into the hf_cache folder in the project directory, and then run app.py.
