"""
AudioRAG Enterprise - Main Application with Authentication

Enhanced Streamlit app with login, analytics, and enterprise features.
"""

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

# Enterprise imports
try:
    from config import settings, CONFIG
    from database.connection import init_database, db_session
    from auth.models import User, RoleType
    from auth.authentication import (
        create_token_pair, authenticate_user, get_password_hash, verify_token
    )
    from auth.authorization import has_permission, is_admin
    from analytics.metrics import get_metrics, track_audio_upload, track_query
    from audit.logger import AuditLogger, AuditAction, init_audit_logger
    from monitoring.health import get_health_status
    ENTERPRISE_ENABLED = True
except ImportError as e:
    logging.warning(f"Enterprise modules not available: {e}")
    ENTERPRISE_ENABLED = False
    from config import CONFIG

# Configure logging
logging.basicConfig(
    filename='rag_audio.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===================================
# Authentication Functions
# ===================================

def show_login_page():
    """Display login page."""
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 100px auto;
        padding: 40px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    .login-title {
        color: white;
        text-align: center;
        font-size: 2rem;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("## 🎶 AudioRAG Enterprise")
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username or Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", use_container_width=True)
                
                if submitted:
                    if username and password:
                        # For demo, accept any login
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.user_role = "analyst"
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Please enter username and password")
        
        with tab2:
            with st.form("register_form"):
                new_email = st.text_input("Email")
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                register_submitted = st.form_submit_button("Register", use_container_width=True)
                
                if register_submitted:
                    if new_password != confirm_password:
                        st.error("Passwords don't match")
                    elif len(new_password) < 8:
                        st.error("Password must be at least 8 characters")
                    else:
                        st.success("Registration successful! Please login.")
        
        st.markdown("---")
        st.markdown("*Powered by SambaNova AI*")


def check_authentication():
    """Check if user is authenticated."""
    if not ENTERPRISE_ENABLED:
        return True  # Skip auth if enterprise not enabled
    
    return st.session_state.get("authenticated", False)


def logout():
    """Log out current user."""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.user_role = None
    st.rerun()


# ===================================
# Original Functions (unchanged)
# ===================================

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
        "authenticated": False,
        "username": None,
        "user_role": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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


def validate_api_keys():
    """Validate required API keys."""
    assemblyai_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not assemblyai_key:
        logger.error("AssemblyAI API key not found")
        st.error("⚠️ ASSEMBLYAI_API_KEY not set in .env file.")
        st.stop()
    return assemblyai_key


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


def format_file_size(size_bytes):
    """Format file size in human-readable format."""
    for unit, threshold in [('', 1024), ('KB', 1024**2), ('MB', 1024**3)]:
        if size_bytes < threshold:
            return f"{size_bytes:.1f} {unit}" if unit else f"{size_bytes} B"
        size_bytes /= 1024
    return f"{size_bytes:.1f} GB"


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


def process_audio_file(uploaded_file, file_path):
    """Process audio file and set up RAG pipeline."""
    try:
        file_key = f"{st.session_state.id}-{uploaded_file.name}"
        st.session_state.current_file_key = file_key

        qdrant_available = check_qdrant_availability()
        st.session_state.qdrant_available = qdrant_available
        if not qdrant_available:
            st.warning(f"Qdrant not available at {CONFIG['qdrant_url']}. Processing may be affected.")

        if file_key not in st.session_state.file_cache:
            with st.status("Processing audio...", expanded=True) as status:
                validate_api_keys()
                status.update(label="Transcribing audio...", state="running")
                transcriber = Transcribe(api_key=os.getenv("ASSEMBLYAI_API_KEY"))
                segments = transcriber.transcribe_audio(file_path)
                st.session_state.transcripts = segments

                unique_speakers = set(s.get('speaker') for s in segments if s.get('speaker') and s.get('confidence', 0) >= CONFIG["min_speaker_confidence"])
                num_speakers = len(unique_speakers)
                
                if not segments:
                    st.error("No transcription data received. Check audio file.")
                    return None

                status.update(label="Preparing documents...", state="running")
                documents = [
                    f"[{int(t['start_time']//60):02d}:{int(t['start_time']%60):02d}] {t.get('speaker', 'Unknown')}: {t['text']}"
                    for t in segments
                ]
                
                status.update(label="Generating embeddings...", state="running")
                embeddata = EmbedData(
                    embed_model_name=CONFIG["embed_model_name"],
                    batch_size=CONFIG["embed_batch_size"]
                )
                embeddata.embed(documents)
                
                status.update(label="Setting up vector database...", state="running")
                vector_db = QdrantVDB_QB(
                    collection_name=f"{CONFIG['collection_name']}_{st.session_state.id}",
                    batch_size=CONFIG["qdrant_batch_size"],
                    vector_dim=CONFIG["vector_dim"]
                )
                if not vector_db.define_client():
                    raise RuntimeError(f"Failed to connect to Qdrant")
                if not vector_db.create_collection():
                    raise RuntimeError("Failed to create vector collection")
                if not vector_db.ingest_data(embeddata):
                    raise RuntimeError("Failed to ingest data")
                
                status.update(label="Initializing RAG system...", state="running")
                retriever = Retriever(vector_db=vector_db, embeddata=embeddata, top_k=5, score_threshold=0.2)
                query_engine = RAG(retriever=retriever, llm_name=CONFIG["llm_name"])
                st.session_state.file_cache[file_key] = query_engine
                
                st.session_state.audio_metadata = {
                    "filename": uploaded_file.name,
                    "file_size": format_file_size(uploaded_file.size),
                    "duration": f"{segments[-1]['end_time']:.1f}s" if segments else "0.0s",
                    "num_speakers": num_speakers,
                    "num_segments": len(segments),
                }
                
                # Track analytics
                if ENTERPRISE_ENABLED:
                    duration = segments[-1]['end_time'] if segments else 0
                    track_audio_upload(
                        file_size_mb=uploaded_file.size / 1024 / 1024,
                        duration_seconds=duration,
                        user_id=st.session_state.get("username"),
                    )
                
                status.update(label="Processing completed!", state="complete")
                logger.info(f"Processed {file_key}: {len(documents)} documents")

        return st.session_state.file_cache[file_key]

    except Exception as e:
        logger.error(f"Processing failed for {uploaded_file.name}: {e}")
        st.error(f"❌ Error processing audio: {str(e)}")
        return None


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
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        background-color: #f8f9fa;
        color: #333333 !important;
    }
    .chat-message.user { background-color: #e3f2fd; border-left-color: #2196f3; }
    .chat-message.assistant { background-color: #f1f8e9; border-left-color: #4caf50; }
    .user-badge {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)


def display_chat_interface():
    """Display the main chat interface."""
    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        st.markdown('<h1 class="main-header">🎶 Audio AI Agent</h1>', unsafe_allow_html=True)
    with col2:
        if ENTERPRISE_ENABLED and st.session_state.get("username"):
            st.markdown(f'<span class="user-badge">👤 {st.session_state.username}</span>', unsafe_allow_html=True)
    with col3:
        if ENTERPRISE_ENABLED:
            if st.button("🚪 Logout"):
                logout()

    # Chat messages
    for message in st.session_state.messages:
        role_class = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
            st.markdown(f'<div class="chat-message {role_class}">{message["content"]}</div>', unsafe_allow_html=True)

    # Chat input
    if st.session_state.current_file_key:
        if prompt := st.chat_input("Ask about the audio conversation..."):
            start_time = time.time()
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
                        
                        # Track query
                        if ENTERPRISE_ENABLED:
                            latency = (time.time() - start_time) * 1000
                            track_query(latency_ms=latency, user_id=st.session_state.get("username"))
                            
                    except Exception as e:
                        st.error(f"❌ Error generating response: {str(e)}")
    else:
        st.info("👆 Please upload an audio file to start chatting!")


def display_sidebar():
    """Display the sidebar with file upload and settings."""
    with st.sidebar:
        st.markdown('<h2>📁 Upload Audio</h2>', unsafe_allow_html=True)
        
        # Show system health
        if ENTERPRISE_ENABLED:
            with st.expander("🔧 System Status"):
                try:
                    health = get_health_status()
                    status_color = "🟢" if health.status.value == "healthy" else "🟡" if health.status.value == "degraded" else "🔴"
                    st.markdown(f"**Status:** {status_color} {health.status.value}")
                    st.markdown(f"**Uptime:** {health.uptime_seconds:.0f}s")
                except Exception as e:
                    st.markdown("**Status:** ⚪ Unknown")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=CONFIG["supported_formats"],
            help=f"Supported: {', '.join(CONFIG['supported_formats'])}. Max: {CONFIG['max_file_size_mb']}MB"
        )

        if uploaded_file:
            is_valid, error_msg = validate_file(uploaded_file)
            if not is_valid:
                st.error(error_msg)
                st.stop()

            st.info(f"📊 **File:** {uploaded_file.name}\n**Size:** {format_file_size(uploaded_file.size)}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name

            try:
                query_engine = process_audio_file(uploaded_file, temp_file_path)
                if query_engine:
                    st.success("🎉 Ready to chat!")
                    st.audio(temp_file_path)
                    display_transcript(st.session_state.transcripts)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        st.markdown("---")
        
        # Analytics (Enterprise)
        if ENTERPRISE_ENABLED:
            with st.expander("📊 Analytics"):
                try:
                    metrics = get_metrics()
                    data = metrics.get_dashboard_data()
                    st.metric("Queries (24h)", int(data.get("queries_24h", {}).get("count", 0)))
                    st.metric("Active Users", data.get("active_users", 0))
                except Exception:
                    st.info("Analytics loading...")
        
        with st.expander("⚙️ Settings"):
            st.slider("Retrieval K", 1, 10, 5)
            st.slider("Score Threshold", 0.0, 1.0, 0.2, 0.1)
        
        st.button("Clear Chat ↺", on_click=reset_chat)


def main():
    """Main application function."""
    st.set_page_config(
        page_title="AudioRAG Enterprise",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    load_dotenv()
    initialize_session_state()
    
    # Check authentication
    if ENTERPRISE_ENABLED and not check_authentication():
        show_login_page()
        return
    
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
