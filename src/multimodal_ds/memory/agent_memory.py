import logging
import json
import uuid
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def _parse_ts(ts_str: str) -> float:
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)  # handle legacy naive ts
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


class AgentMemory:
    def __init__(self, collection_name: str = "agent_memory", ttl_seconds: int = 86400):
        self.collection_name = collection_name
        self.ttl_seconds = ttl_seconds
        self._client = None
        self._collection = None
        self._init_chroma()

    def _init_chroma(self):
        try:
            import chromadb
            from chromadb.config import Settings
            from multimodal_ds.config import OUTPUT_DIR, EMBED_MODEL
            
            persist_directory = str(OUTPUT_DIR / ".chroma_db")
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            collection_name = f"{self.collection_name}"
            
            # --- Dimension Guard ---
            # Get current model dimension via a test embedding
            test_embed = self._get_embedding("dimension_check")
            current_dim = len(test_embed) if test_embed else None
            
            try:
                # Check if collection exists and matches current config
                coll = self._client.get_collection(name=collection_name)
                existing_meta = coll.metadata or {}
                stored_dim = existing_meta.get("embedding_dimension")
                stored_model = existing_meta.get("model_name")
                
                # Reset if dimension or model has changed
                mismatch = False
                if current_dim and stored_dim and str(stored_dim) != str(current_dim):
                    logger.warning(f"[Memory] Dimension mismatch: stored={stored_dim}, current={current_dim}. Resetting.")
                    mismatch = True
                elif stored_model and stored_model != EMBED_MODEL:
                    logger.warning(f"[Memory] Model mismatch: stored={stored_model}, current={EMBED_MODEL}. Resetting.")
                    mismatch = True
                
                if mismatch:
                    self._client.delete_collection(name=collection_name)
            except Exception:
                # Collection doesn't exist yet
                pass

            # Create/Recreate with current metadata
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_dimension": str(current_dim) if current_dim else "unknown",
                    "model_name": EMBED_MODEL
                }
            )
            # ------------------------
            
            logger.info(f"[Memory] ChromaDB collection '{collection_name}' initialized at {persist_directory}")
        except Exception as e:
            logger.warning(f"[Memory] ChromaDB init failed: {e}")
            self._collection = None

    def store(self, content: str, metadata: dict = None, doc_id: str = None) -> str:
        entry_id = doc_id or str(uuid.uuid4())
        # Include timestamp for TTL handling
        from datetime import datetime, timezone
        meta = {"timestamp": datetime.now(timezone.utc).isoformat(), **(metadata or {})}
        meta = {k: str(v) for k, v in meta.items()}
        if self._collection:
            try:
                embedding = self._get_embedding(content)
                self._collection.upsert(
                    ids=[entry_id], documents=[content],
                    embeddings=[embedding] if embedding else None, metadatas=[meta]
                )
            except Exception as e:
                logger.warning(f"[Memory] Store failed: {e}")
        # After inserting, optionally purge old entries
        self._purge_expired()
        return entry_id

    def retrieve(self, query: str, n_results: int = 5, where: dict = None) -> list:
        if not self._collection:
            return []
        try:
            embedding = self._get_embedding(query)
            count = self._collection.count()
            if count == 0:
                return []
            kwargs = {"n_results": min(n_results, count)}
            if embedding:
                kwargs["query_embeddings"] = [embedding]
            else:
                kwargs["query_texts"] = [query]
            if where:
                kwargs["where"] = {"$and": [{k: v} for k, v in where.items()]} if len(where) > 1 else where
            results = self._collection.query(**kwargs)
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            # Filter out expired entries based on timestamp TTL
            filtered = []
            cutoff = datetime.now(timezone.utc).timestamp() - self.ttl_seconds
            for d, m in zip(docs, metas):
                ts_str = m.get("timestamp")
                ts = _parse_ts(ts_str)
                if ts >= cutoff:
                    filtered.append({"content": d, "metadata": m})
            return filtered
        except Exception as e:
            logger.warning(f"[Memory] Retrieve failed: {e}")
            return []

    def store_analysis_step(self, step_name: str, result: str, session_id: str = "default"):
        return self.store(
            content=f"[Step: {step_name}]\n{result}",
            metadata={"step": step_name, "session_id": session_id, "type": "analysis_step"}
        )

    def get_session_history(self, session_id: str) -> list:
        return self.retrieve(query="analysis step result", n_results=20, where={"session_id": session_id})

    def _get_embedding(self, text: str):
        try:
            import httpx
            from multimodal_ds.config import OLLAMA_BASE_URL, EMBED_MODEL
            model_name = EMBED_MODEL.replace("ollama/", "")
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": model_name, "prompt": text[:2000]},
                timeout=30,
            )
            if response.status_code == 200:
                embedding = response.json().get("embedding")
                if embedding:
                    # Log dimension on first call for debugging
                    logger.debug(f"[Memory] Embedding dim={len(embedding)} model={model_name}")
                return embedding
        except Exception:
            pass
        return None

    def count(self) -> int:
        """Return the number of stored memory entries."""
        if not self._collection:
            return 0
        try:
            return self._collection.count()
        except Exception as e:
            logger.warning(f"[Memory] Count failed: {e}")
            return 0


    def get_relevant_lessons(self, query: str, task_type: str = None) -> list:
        """Retrieve up to 10 deduplicated lesson strings for a query.
        If task_type is provided, filters by that task; otherwise returns any reflection.
        """
        try:
            where = {"type": "reflection"}
            if task_type is not None:
                where = {"type": "reflection", "task": task_type}
            results = self.retrieve(query, n_results=10, where=where)
            lessons = []
            seen = set()
            for r in results:
                content = r.get("content", "")
                if "REFLECTION" not in content:
                    continue
                # Extract JSON after the marker
                if "REFLECTION [" in content:
                    try:
                        json_str = content.split("REFLECTION [")[1].split("]: ", 1)[1]
                    except Exception:
                        json_str = content
                else:
                    json_str = content

                # Defensive JSON parsing
                try:
                    parsed = json.loads(json_str)
                except (json.JSONDecodeError, TypeError):
                    # Fallback: use raw content if JSON parsing fails
                    parsed = {"lessons": [json_str]}
                
                if not isinstance(parsed, dict):
                    parsed = {"lessons": [str(parsed)]}

                for lesson in parsed.get("lessons", []):
                    if isinstance(lesson, str) and lesson not in seen:
                        seen.add(lesson)
                        lessons.append(lesson)
                        if len(lessons) >= 10:
                            break
                if len(lessons) >= 10:
                    break
            return lessons
        except Exception as e:
            logger.warning(f"[Memory] get_relevant_lessons failed: {e}")
            return []

    def _purge_expired(self):
        """Delete entries older than TTL from the Chroma collection."""
        if not self._collection:
            return
        try:
            # Retrieve all ids and metadatas
            all_entries = self._collection.get(include=["metadatas"])  # returns dict with keys 'ids' and 'metadatas'
            ids = all_entries.get("ids", [])
            metas = all_entries.get("metadatas", [])
            if not ids:
                return
            cutoff = datetime.now(timezone.utc).timestamp() - self.ttl_seconds
            to_delete = []
            for entry_id, meta in zip(ids, metas):
                ts_str = meta.get("timestamp")
                ts = _parse_ts(ts_str)
                if ts < cutoff:
                    to_delete.append(entry_id)
            if to_delete:
                self._collection.delete(ids=to_delete)
                logger.info(f"[Memory] Purged {len(to_delete)} expired entries")
        except Exception as e:
            logger.warning(f"[Memory] Purge expired failed: {e}")

