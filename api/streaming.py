"""
Streaming Audio API

Real-time audio streaming and processing.
"""

import logging
import asyncio
from datetime import datetime
from typing import AsyncGenerator, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    """A chunk of streaming data."""
    type: str  # "audio", "transcript", "embedding"
    data: Any
    timestamp: str
    sequence: int
    is_final: bool = False


class AudioStreamer:
    """
    Handle real-time audio streaming for live transcription.
    
    Supports WebSocket connections for bidirectional streaming.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100,
    ):
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self._buffer = bytearray()
        self._sequence = 0
        self._is_active = False
    
    async def process_audio_chunk(
        self,
        audio_bytes: bytes,
    ) -> Optional[StreamChunk]:
        """
        Process an incoming audio chunk.
        
        Args:
            audio_bytes: Raw audio bytes
            
        Returns:
            StreamChunk if buffer is ready for processing
        """
        self._buffer.extend(audio_bytes)
        
        # Check if we have enough data
        if len(self._buffer) >= self.chunk_size * 2:  # 16-bit audio = 2 bytes per sample
            chunk_data = bytes(self._buffer[:self.chunk_size * 2])
            self._buffer = self._buffer[self.chunk_size * 2:]
            
            self._sequence += 1
            
            return StreamChunk(
                type="audio",
                data=chunk_data,
                timestamp=datetime.utcnow().isoformat(),
                sequence=self._sequence,
            )
        
        return None
    
    async def stream_transcription(
        self,
        audio_generator: AsyncGenerator[bytes, None],
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream audio and yield transcription chunks.
        
        Args:
            audio_generator: Async generator of audio bytes
            
        Yields:
            StreamChunk with transcription data
        """
        self._is_active = True
        transcript_buffer = ""
        
        try:
            async for audio_bytes in audio_generator:
                chunk = await self.process_audio_chunk(audio_bytes)
                
                if chunk:
                    # Here we would send to AssemblyAI streaming API
                    # For now, yield placeholder
                    yield StreamChunk(
                        type="transcript",
                        data={"text": "[streaming transcription]", "is_partial": True},
                        timestamp=datetime.utcnow().isoformat(),
                        sequence=chunk.sequence,
                    )
            
            # Final chunk
            yield StreamChunk(
                type="transcript",
                data={"text": transcript_buffer, "is_partial": False},
                timestamp=datetime.utcnow().isoformat(),
                sequence=self._sequence + 1,
                is_final=True,
            )
            
        finally:
            self._is_active = False
    
    def stop(self):
        """Stop streaming."""
        self._is_active = False


class StreamingRAG:
    """
    Real-time RAG with streaming responses.
    
    Provides low-latency responses as they're generated.
    """
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
    
    async def stream_query(
        self,
        query: str,
        session_id: str,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream RAG response for a query.
        
        Args:
            query: User query
            session_id: Session ID for context
            
        Yields:
            StreamChunk with response tokens
        """
        # Retrieve context
        context_chunks = self.retriever.search(query)
        context = "\n".join([c.payload["context"] for c in context_chunks])
        
        # Build prompt
        prompt = f"""Context:
{context}

Query: {query}

Answer:"""
        
        # Stream LLM response
        sequence = 0
        full_response = ""
        
        async for token in self._stream_llm(prompt):
            sequence += 1
            full_response += token
            
            yield StreamChunk(
                type="response",
                data={"token": token, "full_text": full_response},
                timestamp=datetime.utcnow().isoformat(),
                sequence=sequence,
            )
        
        # Final chunk
        yield StreamChunk(
            type="response",
            data={"token": "", "full_text": full_response},
            timestamp=datetime.utcnow().isoformat(),
            sequence=sequence + 1,
            is_final=True,
        )
    
    async def _stream_llm(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream tokens from LLM."""
        try:
            response = self.llm.stream_complete(prompt)
            for chunk in response:
                if hasattr(chunk, 'delta'):
                    yield chunk.delta
                elif hasattr(chunk, 'text'):
                    yield chunk.text
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            yield f"[Error: {str(e)}]"


# ===================================
# WebSocket Handler (FastAPI)
# ===================================

async def websocket_handler(websocket, session_id: str):
    """
    Handle WebSocket connection for real-time audio.
    
    Protocol:
    1. Client sends audio chunks (binary)
    2. Server sends transcription updates (JSON)
    3. Client can send queries (JSON with type: "query")
    4. Server streams responses (JSON)
    """
    streamer = AudioStreamer()
    
    try:
        await websocket.accept()
        logger.info(f"WebSocket connected: {session_id}")
        
        while True:
            # Receive message
            message = await websocket.receive()
            
            if "bytes" in message:
                # Audio data
                chunk = await streamer.process_audio_chunk(message["bytes"])
                if chunk:
                    await websocket.send_json({
                        "type": "transcript",
                        "sequence": chunk.sequence,
                        "timestamp": chunk.timestamp,
                    })
            
            elif "text" in message:
                # JSON command
                import json
                data = json.loads(message["text"])
                
                if data.get("type") == "query":
                    # Handle query
                    await websocket.send_json({
                        "type": "response_start",
                        "query": data.get("query"),
                    })
                    
                    # Would stream RAG response here
                    await websocket.send_json({
                        "type": "response_chunk",
                        "text": "Streaming response...",
                    })
                    
                    await websocket.send_json({
                        "type": "response_end",
                    })
                
                elif data.get("type") == "stop":
                    streamer.stop()
                    break
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    
    finally:
        logger.info(f"WebSocket disconnected: {session_id}")
