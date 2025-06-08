import os
import gc
import uuid
import tempfile
import base64
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from qdrant_client import QdrantClient
from rag_code import Transcribe, EmbedData, QdrantVDB_QB, Retriever, RAG
import pkg_resources
from collections import Counter

# Configure logging
logging.basicConfig(
    filename='rag_audio.log',
    level=logging.DEBUG,  # Set to DEBUG for detailed logging
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "collection_name": "chat_with_audios",
    "embed_model_name": "BAAI/bge-large-en-v1.5",
    "llm_name": "Meta-Llama-3.1-405B-Instruct",
    "vector_dim": 1024,
    "embed_batch_size": 32,
    "qdrant_batch_size": 256,
    "max_file_size_mb": 50,
    "qdrant_url": "http://localhost:6333",
    "supported_formats": ["mp3", "wav", "m4a"],
    "min_speaker_confidence": 0.5,  # Lowered to capture more labels
    "min_audio_duration_for_multi_speaker": 30,  # Seconds
}

# Verify llama-index version
try:
    llama_index_version = pkg_resources.get_distribution("llama-index-core").version
    logger.info(f"llama-index-core version: {llama_index_version}")
except Exception as e:
    logger.error(f"Failed to verify llama-index-core version: {e}")

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        "id": uuid.uuid4(),
        "file_cache": {},
        "messages": [],
        "transcripts": [],
        "qdrant_available": None,
        "current_file_key": None,
        "audio_metadata": {},
        "manual_speaker_count": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Reset chat history
def reset_chat():
    """Reset chat history and clean up memory."""
    st.session_state.messages = []
    st.session_state.transcripts = []
    st.session_state.current_file_key = None
    st.session_state.file_cache = {}
    st.session_state.audio_metadata = {}
    st.session_state.manual_speaker_count = 0
    gc.collect()
    logger.info("Chat history and session state reset")
    st.success("Chat history cleared!")

# Validate API keys
def validate_api_keys():
    """Validate required API keys."""
    assemblyai_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not assemblyai_key:
        logger.error("AssemblyAI API key not found")
        st.error("⚠️ ASSEMBLYAI_API_KEY not set in .env file.")
        st.stop()
    return assemblyai_key

# Validate uploaded file
def validate_file(uploaded_file):
    """Validate uploaded file format and size."""
    if not uploaded_file:
        return False, "No file uploaded."
    if uploaded_file.size > CONFIG["max_file_size_mb"] * 1024 * 1024:
        return False, f"File size exceeds {CONFIG['max_file_size_mb']}MB limit."
    ext = Path(uploaded_file.name).suffix.lower()
    if ext.lstrip('.') not in CONFIG["supported_formats"]:
        return False, f"Invalid file format. Use {', '.join(CONFIG['supported_formats'])}."
    return True, None

# Check Qdrant availability
def check_qdrant_availability():
    """Check if Qdrant server is running."""
    try:
        client = QdrantClient(url=CONFIG["qdrant_url"], prefer_grpc=False, timeout=5)
        collections = client.get_collections()
        logger.info(f"Qdrant available at {CONFIG['qdrant_url']}: {collections}")
        return True
    except Exception as e:
        logger.warning(f"Qdrant unavailable at {CONFIG['qdrant_url']}: {e}")
        return False

# Format file size
def format_file_size(size_bytes):
    """Format file size in human-readable format."""
    for unit, threshold in [('', 1024), ('KB', 1024**2), ('MB', 1024**3)]:
        if size_bytes < threshold:
            return f"{size_bytes:.1f} {unit}" if unit else f"{size_bytes} B"
        size_bytes /= 1024
    return f"{size_bytes:.1f} GB"

# Display transcript with timestamps and search
def display_transcript(segments):
    """Display transcript with timestamps, search functionality, and export."""
    st.subheader("📜 Transcript")
    search_term = st.text_input("🔍 Search in transcript:", placeholder="Enter keywords to search")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📜 Export Transcript"):
            transcript_text = "\n".join(
                f"[{int(t['start_time']//60):02d}:{int(t['start_time']%60):02d}] {t['speaker']}: {t['text']}"
                for t in segments
            )
            st.download_button(
                label="Download Transcript",
                data=transcript_text,
                file_name=f"{st.session_state.audio_metadata.get('filename', 'transcript')}.txt",
                mime="text/plain"
            )
    with col2:
        unique_speakers = set(t['speaker'] for t in segments if t.get('speaker') and t.get('confidence', 0) >= CONFIG["min_speaker_confidence"]) if segments else set()
        num_speakers = st.session_state.manual_speaker_count if st.session_state.manual_speaker_count > 0 else len(unique_speakers)
        st.metric("Speakers", num_speakers)
        st.metric("Duration", f"{segments[-1]['end_time']:.1f}s" if segments else "Unknown")
        if num_speakers == 0:
            st.error("No speakers detected. Check audio file or transcription settings.")
        elif num_speakers == 1 and segments and segments[-1]['end_time'] > CONFIG["min_audio_duration_for_multi_speaker"]:
            st.warning("Only one speaker detected in a long audio. Consider setting manual speaker count.")

    with st.expander("Show full transcript", expanded=True):
        for segment in segments:
            timestamp = f"[{int(segment['start_time']//60):02d}:{int(segment['start_time']%60):02d}]"
            text = segment['text']
            highlight_style = ""
            if search_term and search_term.lower() in text.lower():
                text = text.replace(search_term, f"**{search_term}**", 1)
                highlight_style = "background-color: #ffeb3b;"
            speaker = segment.get('speaker', 'Unknown')
            confidence = segment.get('confidence')
            confidence_text = f" (conf: {confidence:.2f})" if confidence is not None else ""
            st.markdown(
                f"""
                <div style="margin-bottom: 8px; padding: 10px; border-left: 3px solid #ddd; {highlight_style}">
                    <strong>{speaker}</strong> <span style="color: #666; font-size: 0.9em;">{timestamp}{confidence_text}</span><br>
                    {text}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Debug: Show speaker distribution and analysis
    with st.expander("🔍 Debug: Speaker Analysis"):
        if segments:
            speaker_counts = Counter(t['speaker'] for t in segments if t.get('speaker'))
            total_segments = len(segments)
            confidences = [t.get('confidence', 0) for t in segments if t.get('confidence') is not None]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            st.markdown(f"**Detected Speakers ({len(unique_speakers)} auto-detected, {num_speakers} displayed):** {', '.join(sorted(unique_speakers))}")
            st.markdown(f"**Manual Speaker Count:** {st.session_state.manual_speaker_count if st.session_state.manual_speaker_count > 0 else 'Not set'}")
            st.markdown("**Speaker Distribution:**")
            for speaker, count in speaker_counts.items():
                percentage = (count / total_segments * 100) if total_segments > 0 else 0
                st.markdown(f"- {speaker}: {count} segments ({percentage:.1f}%)")
            st.markdown(f"**Average Speaker Confidence:** {avg_confidence:.2f}")
            if any(t.get('confidence', 1) < CONFIG["min_speaker_confidence"] for t in segments):
                st.warning(f"Some segments have low-confidence speaker labels (< {CONFIG['min_speaker_confidence']}).")
            # Temporal analysis
            gaps = [segments[i]['start_time'] - segments[i-1]['end_time'] for i in range(1, len(segments)) if segments[i-1]['speaker'] != segments[i]['speaker']]
            avg_gap = sum(gaps) / len(gaps) if gaps else 0
            st.markdown(f"**Average Gap Between Speaker Changes:** {avg_gap:.2f}s")
        else:
            st.markdown("No segments found in the transcript.")

# Process audio file
def process_audio_file(uploaded_file, file_path):
    """Process audio file and set up RAG pipeline."""
    try:
        file_key = f"{st.session_state.id}-{uploaded_file.name}"
        st.session_state.current_file_key = file_key

        # Check Qdrant
        qdrant_available = check_qdrant_availability()
        st.session_state.qdrant_available = qdrant_available
        if not qdrant_available:
            st.warning(f"Qdrant not available at {CONFIG['qdrant_url']}. Processing may be affected.")
            logger.warning("Proceeding with Qdrant check")

        # Transcription
        if file_key not in st.session_state.file_cache:
            with st.status("Processing audio...", expanded=True) as status:
                validate_api_keys()
                status.update(label="Transcribing audio...", state="running")
                transcriber = Transcribe(api_key=os.getenv("ASSEMBLYAI_API_KEY"))
                segments = transcriber.transcribe_audio(file_path)
                st.session_state.transcripts = segments

                # Validate speaker labels
                unique_speakers = set(s.get('speaker') for s in segments if s.get('speaker') and s.get('confidence', 0) >= CONFIG["min_speaker_confidence"])
                num_speakers = len(unique_speakers)
                speaker_counts = Counter(s.get('speaker') for s in segments if s.get('speaker'))
                logger.info(f"Transcription for {uploaded_file.name}: {len(segments)} segments, {num_speakers} unique speakers: {unique_speakers}")
                logger.info(f"Speaker distribution: {dict(speaker_counts)}")
                if not segments:
                    st.error("No transcription data received. Check audio file.")
                    return None
                if num_speakers == 0:
                    logger.error("No speakers detected in transcript.")
                    st.error("No speakers detected. Transcription may be incomplete.")
                elif num_speakers == 1 and segments and segments[-1]['end_time'] > CONFIG["min_audio_duration_for_multi_speaker"]:
                    logger.warning("Only one speaker detected in audio longer than 30s.")
                    st.warning("Only one speaker detected in a long audio. Set manual speaker count if incorrect.")

                # Fallback heuristic: Assume 2 speakers for dialogues >30s if only 1 detected
                if num_speakers == 1 and segments and segments[-1]['end_time'] > CONFIG["min_audio_duration_for_multi_speaker"]:
                    num_speakers = 2 if st.session_state.manual_speaker_count == 0 else st.session_state.manual_speaker_count
                    logger.info(f"Fallback: Assuming {num_speakers} speakers for audio > {CONFIG['min_audio_duration_for_multi_speaker']}s")

                # Prepare documents
                status.update(label="Preparing documents...", state="running")
                documents = [
                    f"[{int(t['start_time']//60):02d}:{int(t['start_time']%60):02d}] {t.get('speaker', 'Unknown')}: {t['text']}"
                    for t in segments
                ]
                # Embeddings
                status.update(label="Generating embeddings...", state="running")
                embeddata = EmbedData(
                    embed_model_name=CONFIG["embed_model_name"],
                    batch_size=CONFIG["embed_batch_size"]
                )
                embeddata.embed(documents)
                # Vector DB
                status.update(label="Setting up vector database...", state="running")
                vector_db = QdrantVDB_QB(
                    collection_name=f"{CONFIG['collection_name']}_{st.session_state.id}",
                    batch_size=CONFIG["qdrant_batch_size"],
                    vector_dim=CONFIG["vector_dim"]
                )
                if not vector_db.define_client():
                    raise RuntimeError(f"Failed to connect to Qdrant at {CONFIG['qdrant_url']}")
                if not vector_db.create_collection():
                    raise RuntimeError("Failed to create vector collection")
                if not vector_db.ingest_data(embeddata):
                    raise RuntimeError("Failed to ingest data into vector store")
                # RAG setup
                status.update(label="Initializing RAG system...", state="running")
                retriever = Retriever(
                    vector_db=vector_db,
                    embeddata=embeddata,
                    top_k=5,
                    score_threshold=0.2
                )
                query_engine = RAG(
                    retriever=retriever,
                    llm_name=CONFIG["llm_name"]
                )
                st.session_state.file_cache[file_key] = query_engine
                # Metadata
                st.session_state.audio_metadata = {
                    "filename": uploaded_file.name,
                    "file_size": format_file_size(uploaded_file.size),
                    "duration": f"{segments[-1]['end_time']:.1f}s" if segments else "0.0s",
                    "num_speakers": num_speakers if st.session_state.manual_speaker_count == 0 else st.session_state.manual_speaker_count,
                    "num_segments": len(segments),
                    "processing_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "speaker_distribution": dict(speaker_counts)
                }
                status.update(label="Processing completed!", state="complete")
                logger.info(f"Processed {file_key}: {len(documents)} documents, {num_speakers} speakers")

        return st.session_state.file_cache[file_key]

    except Exception as e:
        logger.error(f"Processing failed for {uploaded_file.name}: {e}")
        st.error(f"❌ Error processing audio: {str(e)}")
        return None

# Streamlit UI
def create_ui():
    """Create enhanced UI with custom styling."""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1.5rem;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e7d32;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        background-color: #f8f9fa;
        color: #333333 !important;
    }
    .chat-message.user {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .chat-message.assistant {
        background-color: #f1f8e9;
        border-left-color: #4caf50;
    }
    .stChatInput input {
        background-color: #ffffff !important;
        color: #333333 !important;
        border: 1px solid #cccccc !important;
        border-radius: 0.5rem !important;
    }
    .stChatInput input::placeholder {
        color: #666666 !important;
    }
    .stProgress .st-bo {
        background-color: #667eea;
    }
    </style>
    """, unsafe_allow_html=True)

def display_chat_interface():
    """Display the main chat interface."""
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown('<h1 class="main-header">🎶 Audio AI Agent</h1>', unsafe_allow_html=True)
        try:
            st.markdown(
                """
                Powered by <img src="data:image/png;base64,{}" width="150" style="vertical-align:middle; padding-right:10px;">
                and <img src="data:image/png;base64,{}" width="150" style="vertical-align:middle;">
                """.format(
                    base64.b64encode(open("assets/Assemblyai.png", "rb").read()).decode(),
                    base64.b64encode(open("assets/sambanova.png", "rb").read()).decode()
                ),
                unsafe_allow_html=True,
            )
        except FileNotFoundError as e:
            logger.warning(f"Logo images not found: {e}")
            st.markdown("*Powered by AssemblyAI and SambaNova*")

    with col2:
        st.button("Clear Chat ↺", on_click=reset_chat, help="Clear chat history")

    # Initialize chat history
    for message in st.session_state.messages:
        role_class = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
            st.markdown(f'<div class="chat-message {role_class}">{message["content"]}</div>', unsafe_allow_html=True)

    # Chat input
    if st.session_state.current_file_key:
        if prompt := st.chat_input("Ask about the audio conversation..."):
            logger.info(f"User prompt: {prompt}")
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑"):
                st.markdown(f'<div class="chat-message user">{prompt}</div>', unsafe_allow_html=True)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Generating response..."):
                    message_placeholder = st.empty()
                    full_response = ""
                    try:
                        query_engine = st.session_state.file_cache[st.session_state.current_file_key]
                        streaming_response = query_engine.query(prompt)
                        for chunk in streaming_response:
                            try:
                                new_text = chunk.delta if hasattr(chunk, 'delta') else chunk.raw["choices"][0]["delta"].get("content", "")
                                if new_text:
                                    full_response += new_text
                                    message_placeholder.markdown(f'<div class="chat-message assistant">{full_response} ▌</div>', unsafe_allow_html=True)
                            except (KeyError, AttributeError):
                                continue
                        message_placeholder.markdown(f'<div class="chat-message assistant">{full_response}</div>', unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        logger.info(f"Assistant response: {full_response}")
                    except Exception as e:
                        logger.error(f"Query failed: {str(e)}")
                        st.error(f"❌ Error generating response: {str(e)}")
    else:
        st.info("👆 Please upload an audio file to start chatting!")

def display_sidebar():
    """Display the sidebar with file upload and settings."""
    with st.sidebar:
        st.markdown('<h2 class="sidebar-header">📁 Upload Audio</h2>', unsafe_allow_html=True)

        # Qdrant status check
        if st.button("Check Qdrant Status"):
            with st.spinner("Checking Qdrant server..."):
                if check_qdrant_availability():
                    st.success(f"Qdrant server is running at {CONFIG['qdrant_url']}")
                    st.session_state.qdrant_available = True
                else:
                    st.error(
                        f"Qdrant server is not running at {CONFIG['qdrant_url']}. "
                        "Start it with: `docker run -d -p 6333:6333 qdrant/qdrant`"
                    )
                    st.session_state.qdrant_available = False

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=CONFIG["supported_formats"],
            help=f"Supported formats: {', '.join(CONFIG['supported_formats'])}. Max size: {CONFIG['max_file_size_mb']}MB"
        )

        if uploaded_file:
            is_valid, error_msg = validate_file(uploaded_file)
            if not is_valid:
                st.error(error_msg)
                st.stop()

            st.info(f"📊 **File Info:**\n- Name: {uploaded_file.name}\n- Size: {format_file_size(uploaded_file.size)}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name

            try:
                query_engine = process_audio_file(uploaded_file, temp_file_path)
                if query_engine:
                    st.success("🎉 Ready to chat with your audio!")
                    st.audio(temp_file_path)
                    display_transcript(st.session_state.transcripts)
                    if st.session_state.audio_metadata:
                        with st.expander("📊 Audio Analysis", expanded=False):
                            metadata = st.session_state.audio_metadata
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("File Size", metadata.get("file_size", "Unknown"))
                                st.metric("Speakers", metadata.get("num_speakers", 0))
                            with col2:
                                st.metric("Duration", metadata.get("duration", "Unknown"))
                                st.metric("Segments", metadata.get("num_segments", 0))
            except Exception as e:
                logger.error(f"Processing failed: {str(e)}")
                st.error(f"❌ Error processing audio: {str(e)}")
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        st.markdown("---")
        with st.expander("⚙️ Settings"):
            top_k = st.slider(
                "Retrieval K",
                min_value=1,
                max_value=10,
                value=5,
                help="Number of relevant segments to retrieve"
            )
            score_threshold = st.slider(
                "Score Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
                help="Minimum relevance score for retrieved segments"
            )
            manual_speakers = st.number_input(
                "Manual Speaker Count (0 for auto-detection)",
                min_value=0,
                max_value=10,
                value=st.session_state.manual_speaker_count,
                step=1,
                help="Override the detected number of speakers if incorrect"
            )
            if manual_speakers != st.session_state.manual_speaker_count:
                st.session_state.manual_speaker_count = manual_speakers
                if manual_speakers > 0 and st.session_state.audio_metadata:
                    st.session_state.audio_metadata["num_speakers"] = manual_speakers
                    logger.info(f"Manually set speaker count to {manual_speakers}")
            if st.session_state.get("current_file_key") and st.session_state.file_cache:
                query_engine = st.session_state.file_cache[st.session_state.current_file_key]
                query_engine.retriever.top_k = top_k
                query_engine.retriever.score_threshold = score_threshold

        st.markdown("---")
        with st.expander("❓ Help & Tips"):
            st.markdown("""
            **How to use:**
            1. Upload an audio file (MP3, WAV, M4A)
            2. Wait for processing to complete
            3. Check the transcript and speaker count
            4. Set manual speaker count if auto-detection is incorrect
            5. Ask questions about the conversation

            **Example questions:**
            - What was the main topic discussed?
            - Who spoke about [topic]?
            - Summarize the key points
            - What did Speaker A say about [subject]?

            **Tips:**
            - Use clear audio with distinct voices for best speaker detection
            - Check the debug section for speaker analysis
            - Manually override speaker count if detection fails
            - Ensure Qdrant is running for optimal performance
            """)

def main():
    """Main application function."""
    st.set_page_config(
        page_title="RAG over Audio",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    load_dotenv()
    initialize_session_state()
    create_ui()
    validate_api_keys()
    display_sidebar()
    display_chat_interface()

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.8em;">
            Built with ❤️ using Streamlit, AssemblyAI, and Qdrant by Hemant Sudarshan
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()