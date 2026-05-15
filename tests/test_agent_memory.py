import pytest
import json
import uuid
from unittest.mock import MagicMock, patch
from multimodal_ds.memory.agent_memory import AgentMemory

def test_get_relevant_lessons_handles_malformed_json():
    """
    Test that get_relevant_lessons gracefully handles malformed JSON strings
    in the database and falls back to treating them as raw strings if possible.
    """
    # Mock the retrieve method to return various types of malformed or unexpected content
    with patch.object(AgentMemory, 'retrieve') as mock_retrieve:
        mock_retrieve.return_value = [
            # 1. Correct JSON
            {"content": 'REFLECTION [test]: {"lessons": ["lesson 1"]}', "metadata": {"type": "reflection"}},
            # 2. Malformed JSON (missing closing brace)
            {"content": 'REFLECTION [test]: {"lessons": ["lesson 2"', "metadata": {"type": "reflection"}},
            # 3. Not JSON at all, but has the marker
            {"content": 'REFLECTION [test]: This is just a string without JSON structure', "metadata": {"type": "reflection"}},
            # 4. JSON but wrong structure (not a dict)
            {"content": 'REFLECTION [test]: ["just", "a", "list"]', "metadata": {"type": "reflection"}},
            # 5. JSON but missing "lessons" key
            {"content": 'REFLECTION [test]: {"some_other_key": "value"}', "metadata": {"type": "reflection"}},
        ]
        
        # Use a mock collection to avoid real ChromaDB initialization issues in some environments
        with patch('chromadb.EphemeralClient'):
            memory = AgentMemory(collection_name="test_collection")
            lessons = memory.get_relevant_lessons("dummy query")
            
            # Verify that we got results and didn't crash
            assert "lesson 1" in lessons
            
            # Check for fallback results from malformed entries
            # Entry 2 falls back to treating the json_str as a lesson
            assert any('{"lessons": ["lesson 2"' in l for l in lessons)
            # Entry 3 falls back to the raw string
            assert any('This is just a string' in l for l in lessons)
            
            assert len(lessons) >= 3

def test_get_relevant_lessons_deduplication():
    with patch.object(AgentMemory, 'retrieve') as mock_retrieve:
        mock_retrieve.return_value = [
            {"content": 'REFLECTION [test]: {"lessons": ["duplicate", "unique 1"]}', "metadata": {"type": "reflection"}},
            {"content": 'REFLECTION [test]: {"lessons": ["duplicate", "unique 2"]}', "metadata": {"type": "reflection"}},
        ]
        with patch('chromadb.EphemeralClient'):
            memory = AgentMemory()
            lessons = memory.get_relevant_lessons("query")
            
            assert lessons.count("duplicate") == 1
            assert "unique 1" in lessons
            assert "unique 2" in lessons

def test_agent_memory_imports_and_uuid():
    """Verify that uuid is available and imports are correct."""
    with patch('chromadb.EphemeralClient'):
        memory = AgentMemory()
        # Verify store uses uuid (which is now at top level)
        with patch.object(memory, '_collection') as mock_coll:
            with patch.object(memory, '_get_embedding', return_value=[0.1]*1536):
                entry_id = memory.store("test content")
                assert entry_id is not None
                # Should be a valid UUID
                uuid.UUID(entry_id)

def test_ttl_timezone_handling():
    """Verify that entries are not incorrectly expired due to timezone mismatches."""
    with patch('chromadb.EphemeralClient'):
        mem = AgentMemory(ttl_seconds=3600)
        # Mock store to return a fixed ID and verify it doesn't expire immediately
        with patch.object(mem, '_get_embedding', return_value=[0.1]*384):
            entry_id = mem.store("test", metadata={"type": "test"})
            results = mem.retrieve("test", n_results=1)
            assert len(results) == 1  # should not be expired immediately
