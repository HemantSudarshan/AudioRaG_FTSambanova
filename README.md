![image](https://github.com/user-attachments/assets/c24b612f-984b-4d48-84c0-10ed5d19207b)
![image](https://github.com/user-attachments/assets/407c2d36-4a0d-4147-ac5b-db6e71efd7bb)
![image](https://github.com/user-attachments/assets/8aa5d0ab-ea91-427f-aa41-7e765cfc336a)
![Ai as Sercive report](https://github.com/user-attachments/assets/d98080d4-d5cb-432d-994e-2e5f8e15529d)



🔍 Features
Audio Ingestion
Supports MP3, WAV, and M4A formats with preprocessing steps like resampling, normalization, and mono conversion.

Transcription & Speaker Diarization
Powered by AssemblyAI for highly accurate transcription and speaker identification in multi-speaker audio.

RAG Framework
Natural language querying of audio content using:

Qdrant for vector-based semantic search.

SambaNova GPT-3.5 for context-aware, generative answers.

Text Chunking & Embedding
Breaks transcripts into semantic chunks, generates embeddings with OpenAI/HuggingFace, and stores them in Qdrant.

User Interface
Built with Streamlit for:

Audio upload

Transcript viewing

Query interface

Debugging tools

Manual Overrides
Customizable diarization controls (e.g., manual speaker count input, review outputs).

Timestamp Filtering
Perform time-based search and generate session logs for streamlined analysis.

📌 Applications
🎙️ Journalism & Media
Transcribe interviews and podcasts with speaker attribution.

⚖️ Legal & Compliance
Convert courtroom or deposition recordings into searchable transcripts.

📞 Customer Support
Analyze customer service calls for insights and QA.

🏢 Corporate Meetings
Summarize conversations and identify follow-up tasks.

🎓 Research & Academia
Search and analyze lecture audio or field interviews with ease.

🛠️ Tech Stack
Component	Technology Used
Programming	Python
Transcription	AssemblyAI
Semantic Search	Qdrant
Embeddings	OpenAI / HuggingFace
LLM Responses	SambaNova GPT-3.5
UI	Streamlit
Audio Processing	Librosa / Pydub
