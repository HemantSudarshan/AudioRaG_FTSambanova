import logging
from qdrant_client import models, QdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
import assemblyai as aai
from typing import List, Dict
from collections import Counter
from pathlib import Path

# Configure logging
logging.basicConfig(
    filename='rag_audio.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def batch_iterate(lst, batch_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

class EmbedData:
    def __init__(self, embed_model_name="BAAI/bge-large-en-v1.5", batch_size=32):
        self.embed_model_name = embed_model_name
        self.batch_size = batch_size
        self.embed_model = self._load_embed_model()
        self.embeddings = []
        logger.info(f"Initialized EmbedData with model {embed_model_name}")

    def _load_embed_model(self):
        try:
            embed_model = HuggingFaceEmbedding(
                model_name=self.embed_model_name,
                trust_remote_code=True,
                cache_folder='./hf_cache'
            )
            return embed_model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def generate_embedding(self, context):
        try:
            embeddings = self.embed_model.get_text_embedding_batch(context)
            if embeddings and len(embeddings[0]) != 1024:
                raise ValueError(f"Invalid embedding dimension: {len(embeddings[0])}. Expected 1024.")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def embed(self, contexts):
        self.contexts = contexts
        for batch_context in batch_iterate(contexts, self.batch_size):
            batch_embeddings = self.generate_embedding(batch_context)
            self.embeddings.extend(batch_embeddings)
        logger.info(f"Embedded {len(self.contexts)} contexts")

class QdrantVDB_QB:
    def __init__(self, collection_name, vector_dim=1024, batch_size=256):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vector_dim = vector_dim
        self.client = None
        logger.info(f"Initialized QdrantVDB_QB for collection {collection_name}")

    def define_client(self):
        try:
            self.client = QdrantClient(url="http://localhost:6333", prefer_grpc=False, timeout=15)
            logger.info("Qdrant client initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            return False

    def create_collection(self):
        try:
            if not self.client.collection_exists(collection_name=self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_dim,
                        distance=models.Distance.COSINE,
                        on_disk=True
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=5,
                        indexing_threshold=0
                    ),
                    quantization_config=models.BinaryQuantization(
                        binary=models.BinaryQuantizationConfig(always_ram=True)
                    ),
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create Qdrant collection: {e}")
            return False

    def ingest_data(self, embeddata):
        try:
            for batch_context, batch_embeddings in zip(
                batch_iterate(embeddata.contexts, self.batch_size),
                batch_iterate(embeddata.embeddings, self.batch_size)
            ):
                self.client.upload_collection(
                    collection_name=self.collection_name,
                    vectors=batch_embeddings,
                    payload=[{"context": context} for context in batch_context]
                )
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=models.OptimizersConfigDiff(indexing_threshold=10000)
            )
            logger.info(f"Ingested data into Qdrant collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to ingest data into Qdrant: {e}")
            return False

class Retriever:
    def __init__(self, vector_db, embeddata, top_k=5, score_threshold=0.2):
        self.vector_db = vector_db
        self.embeddata = embeddata
        self.top_k = top_k
        self.score_threshold = score_threshold
        logger.info(f"Initialized Retriever with top_k={top_k}, score_threshold={score_threshold}")

    def search(self, query):
        try:
            query_embedding = self.embeddata.embed_model.get_query_embedding(query)
            result = self.vector_db.client.search(
                collection_name=self.vector_db.collection_name,
                query_vector=query_embedding,
                limit=self.top_k,
                score_threshold=self.score_threshold,
                search_params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        ignore=False,
                        rescore=True,
                        oversampling=2.0,
                    )
                ),
            )
            logger.info(f"Retrieved {len(result)} results for query: {query}")
            return result
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise

class RAG:
    def __init__(self, retriever, llm_name="Meta-Llama-3.1-405B-Instruct"):
        self.retriever = retriever
        self.llm_name = llm_name
        self.llm = self._setup_llm()
        system_msg = ChatMessage(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant that answers questions about the user's audio transcript.",
        )
        self.messages = [system_msg]
        self.qa_prompt_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context}\n"
            "---------------------\n"
            "Given the context, answer the query concisely. If you don't know the answer, say 'I don't know!'.\n"
            "Query: {query}\n"
            "Answer: "
        )
        logger.info(f"Initialized RAG with LLM: {llm_name}")

    def _setup_llm(self):
        try:
            from llama_index.llms.sambanovasystems import SambaNovaCloud
            return SambaNovaCloud(
                model=self.llm_name,
                temperature=0.7,
                max_tokens=1024,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize SambaNovaCloud: {e}, falling back to OpenAI")
            from llama_index.llms.openai import OpenAI
            Settings.llm = OpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=1024,
            )
            return Settings.llm

    def generate_context(self, query):
        try:
            result = self.retriever.search(query)
            context = [dict(data.payload)["context"] for data in result]
            return "\n\n---\n\n".join(context[:3])
        except Exception as e:
            logger.error(f"Error generating context: {e}")
            raise

    def query(self, query):
        try:
            context = self.generate_context(query=query)
            prompt = self.qa_prompt_tmpl_str.format(context=context, query=query)
            user_msg = ChatMessage(role=MessageRole.USER, content=prompt)
            streaming_response = self.llm.stream_complete(user_msg.content)
            return streaming_response
        except Exception as e:
            logger.error(f"Error during query: {e}")
            raise

class Transcribe:
    def __init__(self, api_key: str):
        try:
            aai.settings.api_key = api_key
            self.transcriber = aai.Transcriber()
            logger.info("Initialized AssemblyAI transcriber")
        except Exception as e:
            logger.error(f"Failed to initialize AssemblyAI: {e}")
            raise

    def transcribe_audio(self, audio_path: str) -> List[Dict[str, any]]:
        try:
            config = aai.TranscriptionConfig(
                speaker_labels=True,
                audio_end_at=3600000,
                disfluencies=False,
                punctuate=True,
            )
            transcript = self.transcriber.transcribe(audio_path, config=config)
            if transcript.status == aai.TranscriptStatus.error:
                raise ValueError(f"Transcription failed: {transcript.error}")

            # Log raw utterance data
            raw_utterances = [
                {
                    "speaker": u.speaker,
                    "text": u.text,
                    "start": u.start / 1000 if hasattr(u, 'start') else 0,
                    "end": u.end / 1000 if hasattr(u, 'end') else 0,
                    "confidence": u.confidence if hasattr(u, 'confidence') else None,
                    "duration": (u.end - u.start) / 1000 if hasattr(u, 'start') and hasattr(u, 'end') else 0
                }
                for u in transcript.utterances
            ]
            logger.debug(f"Raw utterances for {audio_path}: {raw_utterances}")

            # Normalize speaker labels
            speaker_map = {}
            next_speaker_id = 'A'
            utterances = []
            for utterance in transcript.utterances:
                raw_speaker = utterance.speaker
                if raw_speaker not in speaker_map:
                    speaker_map[raw_speaker] = f"Speaker {next_speaker_id}"
                    next_speaker_id = chr(ord(next_speaker_id) + 1)
                utterances.append({
                    "speaker": speaker_map[raw_speaker],
                    "text": utterance.text,
                    "start_time": utterance.start / 1000 if hasattr(utterance, 'start') else 0,
                    "end_time": utterance.end / 1000 if hasattr(utterance, 'end') else 0,
                    "confidence": utterance.confidence if hasattr(utterance, 'confidence') else None
                })

            # Validate speaker distribution
            speaker_counts = Counter(u['speaker'] for u in utterances)
            total_segments = len(utterances)
            dominant_speaker = max(speaker_counts.items(), key=lambda x: x[1], default=(None, 0))
            dominant_percentage = (dominant_speaker[1] / total_segments * 100) if total_segments > 0 else 0
            logger.info(f"Transcribed {audio_path}: {len(utterances)} utterances, {len(speaker_map)} speakers")
            logger.info(f"Speaker distribution: {dict(speaker_counts)}")
            if not utterances:
                logger.warning("No utterances found in transcription")
            elif len(speaker_map) == 1 and total_segments > 10:
                logger.warning(f"Only one speaker detected with {total_segments} segments. Dominant: {dominant_speaker[0]} ({dominant_percentage:.1f}%). Consider manual override.")
            elif dominant_percentage > 90:
                logger.warning(f"Dominant speaker {dominant_speaker[0]} has {dominant_percentage:.1f}% of segments. Possible diarization issue. Consider manual override.")

            return utterances
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise