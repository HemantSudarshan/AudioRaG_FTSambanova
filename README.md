🎧 AudioRAG: Audio-to-Insight AI Agent


📖 Overview
AudioRAG: Audio-to-Insight AI Agent is an innovative AI-powered tool that transforms audio content into actionable insights through transcription, speaker diarization, and natural language querying. Developed by Hemant Sudarshan and V Tharun as part of their B.Tech in Computer Science and Technology at Dayananda Sagar University, this project leverages Retrieval-Augmented Generation (RAG) and SambaNova Systems’ high-performance AI infrastructure to enable seamless audio analytics.

🔍 Features
Audio Ingestion

Supports MP3, WAV, and M4A formats
Preprocessing: resampling, normalization, mono conversion

Transcription & Speaker Diarization

Powered by AssemblyAI
Accurate transcription with multi-speaker identification

RAG Framework

Natural language querying of audio content using:
Qdrant for vector-based semantic search
SambaNova GPT-3.5 for context-aware generative responses



Text Chunking & Embedding

Transcripts segmented into semantic chunks
Embeddings generated via OpenAI / HuggingFace models
Stored in Qdrant for efficient retrieval

User Interface

Built with Streamlit
Features:
Audio upload
Transcript viewing
Query interface
Debugging tools



Manual Overrides

Customize diarization by specifying speaker counts
Review and correct diarization outputs

Timestamp Filtering

Time-based search capability
Session logs for improved analysis


📌 Applications

🎙️ Journalism & MediaTranscribe interviews and podcasts with speaker attribution

⚖️ Legal & ComplianceConvert courtroom or deposition recordings into searchable transcripts

📞 Customer SupportAnalyze call interactions for quality assurance

🏢 Corporate MeetingsSummarize discussions and track action items

🎓 Research & AcademiaQuery and analyze lectures or interviews with ease



🛠️ Tech Stack



Component
Technology Used



Programming
Python


Transcription
AssemblyAI


Semantic Search
Qdrant


Embeddings
OpenAI / HuggingFace


LLM Responses
SambaNova GPT-3.5


User Interface
Streamlit


Audio Processing
Librosa / Pydub



🚀 Installation

Clone the repository:git clone https://github.com/your-repo/audiorag.git
cd audiorag


Install dependencies:pip install -r requirements.txt


Download the embedding model:
Link: https://www.mediafire.com/folder/41tpj4qu5ltyd/hf_cache
Extract and place the model in the project’s hf_cache folder.


Set up environment variables:
Create a .env file in the root directory.
Add API keys:ASSEMBLYAI_API_KEY=your_assemblyai_key
OPENAI_API_KEY=your_openai_key
SAMBANOVA_API_KEY=your_sambanova_key




Run the Streamlit app:streamlit run app.py




🖱️ Usage

Launch the Streamlit app (e.g., http://localhost:8501).
Upload an audio file (MP3, WAV, or M4A).
View the generated transcript with speaker labels.
Query the transcript using natural language (e.g., “What did Speaker 1 say about deadlines?”).
Use manual overrides to correct diarization errors if needed.


📂 Project Links

Embedding Model: https://www.mediafire.com/folder/41tpj4qu5ltyd/hf_cache



🛑 Challenges Overcome

Resolved compatibility issues with audio libraries (librosa, pydub) through code refactoring.
Mitigated AssemblyAI dual_channel transcription errors with fallback mechanisms.
Improved diarization accuracy for complex audio (e.g., overlapping speech) using manual overrides.


🔮 Future Enhancements

Real-time transcription for live audio streams
Sentiment analysis to detect speaker emotions
Multilingual support for global accessibility
Integration with enterprise tools (Slack, Teams)
Mobile app for iOS and Android


👥 Contributors

Hemant Sudarshan - Lead Developer

