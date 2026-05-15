"""
Top-level LangGraph StateGraph - wires all agents as nodes with a
MemorySaver checkpointer for session persistence.

Fixes applied (vs original):
  1. _decide_ingestion_path: returns a single string, not a list.
     Fan-out to multiple ingestion nodes requires Send() - this simpler
     approach routes to the FIRST matching type, which is correct for
     the current sequential graph topology.
  2. _reviewer_node: task_result dict now uses keys that evaluation_agent
     actually reads ("name", "success", "output_preview", "files_created").
  3. retry_count: incremented in state when retrying, preventing infinite loops.
"""
from __future__ import annotations

import logging
from langgraph.types import Send
import json
from datetime import datetime, timezone
from multimodal_ds.config import OUTPUT_DIR
import uuid
from typing import Optional


logger = logging.getLogger(__name__)
from multimodal_ds.agents.evaluation_agent import EvaluationAgent, FLAG_OVERALL_THRESHOLD
from multimodal_ds.agents.code_execution_agent import CodeExecutionAgent
from multimodal_ds.agents.problem_understanding_agent import ProblemUnderstandingAgent
from multimodal_ds.agents.reflection_agent import ReflectionAgent, ReflectionReport
from multimodal_ds.core.context_pool import get_context_pool
session_logger = logging.getLogger('session_log')
if not session_logger.handlers:
    handler = logging.FileHandler(OUTPUT_DIR / 'session_log.jsonl')
    handler.setLevel(logging.INFO)
    # Use raw message (JSON string) without extra formatting
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    session_logger.addHandler(handler)
    session_logger.propagate = False


MAX_RETRIES = 2


def _sanitize_for_checkpoint(data):
    import numpy as np
    if isinstance(data, dict):
        return {k: _sanitize_for_checkpoint(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_sanitize_for_checkpoint(v) for v in data]
    if hasattr(data, "item") and not isinstance(data, (str, bytes)):
        return data.item()
    if isinstance(data, (np.integer, np.floating)):
        return float(data) if isinstance(data, np.floating) else int(data)
    return data


# ── Node functions ───────────────────────────────────────────────────────────

def _router_node(state):
    """Determine routing flags based on uploaded file extensions.
    Returns a dict with boolean flags for each document type.
    """
    from pathlib import Path
    EXTENSIONS = {
        "doc":   {".pdf", ".docx", ".txt", ".md", ".html", ".rst"},
        "image": {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"},
        "audio": {".mp3", ".wav", ".m4a", ".ogg", ".flac"},
        "table": {".csv", ".xlsx", ".parquet", ".json", ".tsv"},
    }
    flags = {k: False for k in EXTENSIONS}
    for path in state.get("uploaded_files", []):
        ext = Path(path).suffix.lower()
        for kind, exts in EXTENSIONS.items():
            if ext in exts:
                flags[kind] = True
    logger.info(f"[Graph/Router] Routing flags: {flags}")
    return {"_routing_flags": flags}
def _decide_ingestion_path(state) -> list[Send]:
    """Determine which ingestion nodes to invoke based on routing flags.
    Returns a LIST of Send() objects for parallel execution.
    """
    flags = state.get("_routing_flags", {})
    node_map = {
        "table": "tab_ingest",
        "doc": "doc_ingest",
        "image": "img_ingest",
        "audio": "audio_ingest"
    }
    targets = [node_map[k] for k, v in flags.items() if v and k in node_map]
    
    # If no ingestion needed, send to merge_ingest to continue the flow
    if not targets:
        return [Send("merge_ingest", state)]
    
    return [Send(t, state) for t in targets]

def _ingest_merge_node(state):
    """Merge ingestion results, perform blocked document checks, and sanitize state."""
    from multimodal_ds.core.message_bus import get_bus, AgentMessage, MessageType
    from multimodal_ds.core.context_pool import SharedContextPool
    import logging
    
    logger = logging.getLogger(__name__)
    session_id = state.get("session_id", "default")
    errors = state.get("errors", [])
    if errors is None:
        errors = []
    
    # 1. Collect all BLOCKED documents from parsed_documents
    parsed_documents = state.get("parsed_documents", [])
    blocked = [d for d in parsed_documents if d.get("status") == "blocked"]
    
    # 2. If any blocked docs found
    if blocked:
        bus = get_bus()
        for d in blocked:
            source = d.get("provenance", {}).get("source_path", "?")
            entity_types = d.get("metadata", {}).get("pii_report", {}).get("entity_types_found", [])
            
            # a. Log at WARNING level
            logger.warning(f"[MergeIngest] BLOCKED document: {source} (Entities: {entity_types})")
            
            # b. Publish INGEST_BLOCKED to MessageBus
            bus.publish(AgentMessage(
                msg_type=MessageType.INGEST_BLOCKED,
                payload={
                    "source": source,
                    "entity_types": entity_types
                },
                sender="ingest_merge",
                session_id=session_id
            ))
            
            # c. Add to state["errors"]
            errors.append(f"[PII BLOCK] {source} blocked: {entity_types}")
            
    state["errors"] = errors
            
    # 3. Build a data_context_summary string from non-blocked tabular summaries
    tabular_summaries = []
    for d in parsed_documents:
        if d.get("status") != "blocked":
            summary = d.get("metadata", {}).get("tabular_summary")
            if summary:
                tabular_summaries.append(str(summary))
                
    if tabular_summaries:
        summary_text = "\n".join(tabular_summaries)
        pool = SharedContextPool()
        pool.set("ingest_summary", summary_text, agent="ingest_merge")

    # 4. Return the updated state with the new errors appended
    return _sanitize_for_checkpoint(state)


def _problem_understanding_node(state):
    """Invoke ProblemUnderstandingAgent to produce a problem spec and store it in state."""
    try:
        agent = ProblemUnderstandingAgent(session_id=state.get("session_id", "default"))
        spec = agent.understand(state.get("user_query", ""), state.get("uploaded_files", []))
        return {"problem_spec": spec.to_dict()}
    except Exception as e:
        logger.error(f"[ProblemUnderstanding] Node error: {e}")
        return {"problem_spec": {}}


def _doc_ingest_node(state):
    from multimodal_ds.ingestion.pdf_ingestion import ingest_pdf
    from multimodal_ds.ingestion.router import _ingest_plain_text
    from pathlib import Path

    DOC_EXTS = {".pdf", ".docx", ".txt", ".md", ".html", ".rst"}
    docs = list(state.get("parsed_documents", []))

    for fp in state.get("uploaded_files", []):
        if Path(fp).suffix.lower() in DOC_EXTS:
            doc = ingest_pdf(fp) if fp.endswith(".pdf") else _ingest_plain_text(fp)
            docs.append(doc.to_dict())

    vector_store_id = state.get("vector_store_id", "")
    text_chunks = [d.get("text_content", "")[:2000] for d in docs if d.get("text_content")]
    if text_chunks:
        try:
            from multimodal_ds.memory.agent_memory import AgentMemory
            mem = AgentMemory(collection_name="doc_chunks")
            for chunk in text_chunks:
                mem.store(chunk, metadata={"type": "document"})
            vector_store_id = str(mem._collection.name) if mem._collection else vector_store_id
        except Exception as e:
            logger.warning(f"[Graph/DocIngest] ChromaDB store failed: {e}")

    return {"parsed_documents": docs, "vector_store_id": vector_store_id}


def _img_ingest_node(state):
    from multimodal_ds.ingestion.image_ingestion import ingest_image, SUPPORTED_IMAGES
    from pathlib import Path

    embeddings = list(state.get("image_embeddings", []))
    for fp in state.get("uploaded_files", []):
        if Path(fp).suffix.lower() in SUPPORTED_IMAGES:
            doc = ingest_image(fp)
            if doc.embeddings:
                embeddings.append(doc.embeddings)
    return {"image_embeddings": embeddings}


def _audio_ingest_node(state):
    from multimodal_ds.ingestion.audio_ingestion import ingest_audio, SUPPORTED_AUDIO
    from pathlib import Path

    transcripts = list(state.get("audio_transcripts", []))
    for fp in state.get("uploaded_files", []):
        if Path(fp).suffix.lower() in SUPPORTED_AUDIO:
            doc = ingest_audio(fp)
            if doc.text_content:
                transcripts.append(doc.text_content)
    return {"audio_transcripts": transcripts}


def _tab_ingest_node(state):
    from multimodal_ds.ingestion.tabular_ingestion import ingest_tabular, SUPPORTED_TABULAR
    from pathlib import Path

    summaries = list(state.get("tabular_summaries", []))
    for fp in state.get("uploaded_files", []):
        if Path(fp).suffix.lower() in SUPPORTED_TABULAR:
            doc = ingest_tabular(fp)
            if doc.schema_info:
                summaries.append({
                    "source":       fp,
                    "shape":        doc.schema_info.get("shape", []),
                    "columns":      doc.schema_info.get("columns", []),
                    "dtypes":       doc.schema_info.get("dtypes", {}),
                    "sample":       doc.text_content[:1500],
                    "data_profile": doc.data_profile,
                })
    return {"tabular_summaries": _sanitize_for_checkpoint(summaries)}


def _model_selection_node(state):
    """Select and configure models based on statistical report and AutoML suggestion."""
    import pandas as pd
    from pathlib import Path
    from multimodal_ds.agents.model_selection_agent import ModelSelectionAgent

    try:
        tab_summaries = state.get("tabular_summaries", [])
        if not tab_summaries:
            logger.info("[ModelSelection] No tabular summaries – skipping.")
            return {}
        first = tab_summaries[0]
        file_path = first.get("source")
        if not file_path:
            logger.info("[ModelSelection] Tabular summary missing source – skipping.")
            return {}

        # Load dataframe based on extension
        df = None
        p = Path(file_path)
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        elif p.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(file_path)
        elif p.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path)

        if df is None:
            logger.warning(f"[ModelSelection] Unsupported file extension for {file_path}")
            return {}

        automl_suggestion = first.get("automl_suggestion", {})
        target_col = automl_suggestion.get("target_candidates", [None])[0]
        stat_report = state.get("statistical_report", {})

        agent = ModelSelectionAgent(session_id=state.get("session_id", "default"))
        result = agent.select_models(df, target_col, stat_report, automl_suggestion)
        # Determine dataset size for optional tuning
        shape = first.get("shape", [0, 0])
        n_rows = shape[0] if isinstance(shape, (list, tuple)) and len(shape) > 0 else 0

        # Generate baseline ensemble code (large-dataset path, no Optuna).
        # generate_ensemble_code is deterministic and does not call an LLM.
        try:
            tuned_code = agent.generate_ensemble_code(result, "df", target_col) or ""
            if not tuned_code:
                logger.warning("[ModelSelection] Code generation skipped for large dataset")
        except Exception as _gen_err:
            logger.warning(
                f"[ModelSelection] Code generation skipped for large dataset: {_gen_err}"
            )
            tuned_code = ""

        tuning_results = {}
        if 0 < n_rows <= 50000:
            try:
                logger.info(f"[ModelSelection] Starting Optuna tuning for {n_rows} rows...")
                # Prepare X and y for tuning
                X = df.drop(columns=[target_col]) if target_col and target_col in df.columns else df
                y = df[target_col] if target_col and target_col in df.columns else None
                if y is not None:
                    tuning_results = agent.tune_all_models(X, y, result)
                    tuned_code = agent.generate_tuned_ensemble_code(result, tuning_results, "df", target_col)
                    logger.info(f"[ModelSelection] Tuning complete for {result.get('primary_model')}")
                else:
                    logger.warning("[ModelSelection] Target column missing for tuning; using default code")
            except Exception as e:
                logger.warning(f"[ModelSelection] Optuna tuning failed: {e} - using default ensemble code")
        else:
            logger.info(f"[ModelSelection] Skipping tuning (n_rows={n_rows})")

        return {"model_selection": result, "tuning_results": tuning_results, "ensemble_code_template": tuned_code}
    except Exception as e:
        logger.warning(f"[ModelSelection] Node error: {e}")
        return {}


def _stats_validation_node(state):
    """Run statistical validation on the dataset."""
    from multimodal_ds.agents.statistical_agent import StatisticalReasoningAgent
    import pandas as pd

    uploaded = state.get("uploaded_files", [])
    tab_file = next((f for f in uploaded if f.endswith((".csv", ".xlsx", ".parquet"))), None)
    if not tab_file:
        return {}

    try:
        df = pd.read_csv(tab_file) if tab_file.endswith(".csv") else pd.read_excel(tab_file)
        agent = StatisticalReasoningAgent(session_id=state.get("session_id", "default"))
        report = agent.validate_dataset(df)
        return {"statistical_report": _sanitize_for_checkpoint(report)}
    except Exception as e:
        logger.warning(f"[Graph/Stats] Validation failed: {e}")
        return {}


def _planner_node(state):
    from multimodal_ds.agents.planner_agent import run_planner
    from pathlib import Path

    # Build a rich data‑context string (numeric stats, missing‑value info) – same as executor
    data_context_parts = []
    for t in state.get("tabular_summaries", [])[:2]:
        cols = t.get("columns", [])
        shape = t.get("shape", [])
        profile = t.get("data_profile", {})
        data_context_parts.append(
            f"Table {Path(t['source']).name}: {shape} rows×cols\n"
            f"Columns: {cols}\n"
        )
        if profile.get("numeric_stats"):
            data_context_parts.append("Numeric column stats (mean / std / min / max):")
            for col, s in list(profile["numeric_stats"].items())[:10]:
                data_context_parts.append(
                    f"  {col}: mean={s.get('mean', 0):.2f}, std={s.get('std', 0):.2f}, "
                    f"min={s.get('min', 0):.2f}, max={s.get('max', 0):.2f}"
                )
        missing = {k: v for k, v in profile.get("missing_values", {}).items() if v > 0}
        if missing:
            data_context_parts.append(f"Missing values: {missing}")
        else:
            data_context_parts.append("Missing values: none detected")
    planner_data_context = "\n".join(data_context_parts) if data_context_parts else ""
    # Store in state for potential downstream use
    state["planner_data_context"] = planner_data_context

    # -----------------------------------------------------------------
    # Run the planner LLM – we provide the user query and any available
    # document profiles (here a minimal empty list, since the graph does not
    # collect UnifiedDocument objects). The planner returns a dict with the
    # analysis plan and tasks.
    # -----------------------------------------------------------------
    from multimodal_ds.core.schema import UnifiedDocument, DataType, ProcessingStatus

    proxy_docs = []

    # 1. Tabular documents
    for t in state.get("tabular_summaries", []):
        doc = UnifiedDocument(
            data_type=DataType.TABULAR,
            status=ProcessingStatus.DONE,
            text_content=t.get("sample", ""),
            schema_info={
                "columns": t.get("columns", []),
                "shape": t.get("shape", []),
                "numeric_cols": [c for c, d in t.get("dtypes", {}).items()
                                 if "int" in d or "float" in d],
            },
            metadata={"automl_suggestion": t.get("automl_suggestion", {})}
        )
        proxy_docs.append(doc)

    # 2. Text/PDF documents (exclude BLOCKED)
    for d in state.get("parsed_documents", []):
        text = d.get("text_content", "")
        if text and not text.startswith("[BLOCKED"):
            doc = UnifiedDocument(
                data_type=DataType.TEXT,
                status=ProcessingStatus.DONE,
                text_content=text[:1500],
            )
            proxy_docs.append(doc)

    # 3. Audio transcripts
    for transcript in state.get("audio_transcripts", []):
        if transcript and not transcript.startswith("[BLOCKED"):
            doc = UnifiedDocument(
                data_type=DataType.AUDIO,
                status=ProcessingStatus.DONE,
                text_content=transcript[:1000],
            )
            proxy_docs.append(doc)

    # 4. Inject statistical report as context
    stat_report = state.get("statistical_report", {})
    stat_context = ""
    if stat_report and isinstance(stat_report, dict):
        non_normal = [k for k, v in stat_report.get("normality", {}).items()
                          if isinstance(v, dict) and not v.get("is_normal", True)]
        n_strong = stat_report.get("correlation", {}).get("n_strong", 0)
        mc = stat_report.get("multicollinearity", {}).get("multicollinearity_detected", False)
        stat_context = (
            f"Statistical findings: non-normal columns={non_normal}, "
            f"strong_correlations={n_strong}, multicollinearity={mc}"
        )
        if stat_context:
            doc = UnifiedDocument(
                data_type=DataType.TEXT,
                status=ProcessingStatus.DONE,
                text_content=f"Statistical Validation Report:\n{stat_context}",
            )
            proxy_docs.append(doc)

    # Include required deliverables from the problem spec (if any) in the planner prompt.
    problem_spec = state.get("problem_spec", {})
    user_objective = state.get("user_query", "")
    required_deliverables = problem_spec.get("required_deliverables", [])
    if required_deliverables:
        user_objective = (
            f"{user_objective}\nDesired deliverables: {', '.join(required_deliverables)}"
        )
    try:
        plan_result = run_planner(
            user_objective=user_objective,
            documents=proxy_docs,
            session_id=state.get("session_id", "default"),
        )
    except Exception as e:
        logger.warning(f"[Planner] run_planner failed: {e}")
        plan_result = {}

    tasks = plan_result.get("analysis_plan", [])
    return {
        "analysis_plan":  plan_result.get("final_plan", ""),
        "analysis_tasks": tasks,
        "hypotheses":     plan_result.get("hypotheses", []),
        "current_step":   0,
        "steps_total":    len(tasks),
    }


def _executor_node(state):
    """
    Execute the current analysis task via CodeExecutionAgent.

    Sequencing:
      1. Guard - if current_step >= len(tasks), return empty dict (no-op).
      2. Build data_context from tabular_summaries (shape, columns, numeric
         stats, missing values) - mirrors _planner_node exactly.
      3. Inject ChromaDB RAG context from AgentMemory("doc_chunks").
      4. Run CodeExecutionAgent(session_id).execute(description, data_context,
         file_paths).
      5. PII-scan every generated file via the module-level get_pii_guard()
         singleton (not a fresh instantiation).
      6. Write structured JSON audit log via session_logger.
      7. Increment telemetry on success.
      8. Return all required AgentState keys.
    """
    from pathlib import Path
    from multimodal_ds.memory.agent_memory import AgentMemory
    from multimodal_ds.core.pii_guard import get_pii_guard
    from multimodal_ds.core.telemetry import get_telemetry

    # ── 1. Guard clause ──────────────────────────────────────────────────────
    tasks = state.get("analysis_tasks", [])
    step = state.get("current_step", 0)
    session_id = state.get("session_id", "default")

    if step >= len(tasks):
        logger.info(f"[Executor] step={step} >= len(tasks)={len(tasks)} - no-op return.")
        return {}

    task = tasks[step]

    # ── 2. Build data_context (same pattern as _planner_node) ────────────────
    data_context_parts = []
    for t in state.get("tabular_summaries", [])[:2]:
        cols = t.get("columns", [])
        shape = t.get("shape", [])
        profile = t.get("data_profile", {})
        data_context_parts.append(
            f"Table {Path(t['source']).name}: {shape} rows×cols\n"
            f"Columns: {cols}\n"
        )
        if profile.get("numeric_stats"):
            data_context_parts.append("Numeric column stats (mean / std / min / max):")
            for col, s in list(profile["numeric_stats"].items())[:10]:
                data_context_parts.append(
                    f"  {col}: mean={s.get('mean', 0):.2f}, std={s.get('std', 0):.2f}, "
                    f"min={s.get('min', 0):.2f}, max={s.get('max', 0):.2f}"
                )
        missing = {k: v for k, v in profile.get("missing_values", {}).items() if v > 0}
        if missing:
            data_context_parts.append(f"Missing values: {missing}")
        else:
            data_context_parts.append("Missing values: none detected")
    data_context = "\n".join(data_context_parts) if data_context_parts else ""

    # ── 3. Inject RAG context from AgentMemory("doc_chunks") ─────────────────
    try:
        mem = AgentMemory(collection_name="doc_chunks")
        rag_results = mem.retrieve(task.get("description", ""), n_results=4)
        if rag_results:
            rag_text = "\n\n".join(
                r["content"] for r in rag_results if r.get("content")
            )
            data_context = (
                f"Relevant document context (from ChromaDB):\n{rag_text}\n\n"
                + data_context
            )
    except Exception as e:
        logger.warning(f"[Executor] RAG retrieval failed: {e}")

    # -- 4. File paths - upload original data files into agent sandbox ---------
    data_files = []
    for fp in state.get("uploaded_files", []):
        p = Path(fp)
        if p.exists():
            data_files.append(str(p.resolve()))  # resolve to absolute path
        else:
            logger.warning(f"[Executor] File not found: {fp}")

    # Also inject the filename into the task description so the LLM
    # knows exactly what to load:
    file_names = [Path(f).name for f in data_files]
    task_description = task.get("description", "")
    if file_names and "read_csv" not in task_description.lower():
        task_description = (
            f"{task_description}\n\n"
            f"Available data files in working directory: {file_names}\n"
            f"Load the primary dataset with: df = pd.read_csv('{file_names[0]}')"
        )

    # ── 5. Execute via CodeExecutionAgent ────────────────────────────────────
    logger.info(f"[Executor] Step {step + 1}/{len(tasks)}: {task.get('name', 'task')}")
    try:
        agent = CodeExecutionAgent(session_id=session_id)
        result = agent.execute(
            task_description=task_description,
            data_context=data_context,
            file_paths=data_files,
        )
    except Exception as e:
        logger.error(f"[Executor] CodeExecutionAgent raised: {e}")
        result = {
            "success": False,
            "output": f"Agent raised exception: {e}",
            "files_created": [],
            "error": str(e),
        }

    success: bool = result.get("success", False)
    full_output: str = result.get("output", "")
    files: list = result.get("files_created", [])
    error_str: str = result.get("error", "")

    # ── 6. PII scan on generated files ───────────────────────────────────────
    guard = get_pii_guard()
    pii_blocked_files = []
    pii_clean_files = []
    for fname in files:
        try:
            fpath = Path(OUTPUT_DIR) / session_id / fname
            if not fpath.exists():
                pii_clean_files.append(fname)
                continue
            if fpath.suffix.lower() in {".txt", ".csv", ".json", ".html", ".md"}:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
                report = guard.scan_text(content, source=fname)
                if report.blocked:
                    logger.warning(f"[Executor] PII BLOCKED in generated file '{fname}' - omitting.")
                    pii_blocked_files.append(fname)
                else:
                    pii_clean_files.append(fname)
            else:
                pii_clean_files.append(fname)
        except Exception as pii_err:
            logger.warning(f"[Executor] PII scan failed for '{fname}': {pii_err} - passing file")
            pii_clean_files.append(fname)

    # Use only PII-clean files for downstream state
    files = pii_clean_files

    # ── 7. Partition files by type ────────────────────────────────────────────
    image_exts = {".png", ".jpg", ".webp"}
    visualizations = [f for f in files if Path(f).suffix.lower() in image_exts]
    saved_artifacts = [f for f in files if Path(f).suffix.lower() not in image_exts]

    # ── 8. Build truncated code output for code_outputs (summary field) ───────
    truncated_output = full_output[:2000] + ("...[truncated]" if len(full_output) > 2000 else "")
    errors_out = [f"Step {step + 1}: {error_str}"] if error_str and not success else []

    # ── 9. Structured JSON audit log ──────────────────────────────────────────
    log_entry = {
        "event":        "executor_step",
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "session_id":   session_id,
        "step":         step,
        "step_name":    task.get("name", f"step_{step}"),
        "success":      success,
        "files_created": files,
        "pii_blocked":  pii_blocked_files,
        "output_chars": len(full_output),
        "error":        error_str if not success else "",
    }
    session_logger.info(json.dumps(log_entry))

    # ── 10. Telemetry increment on success ────────────────────────────────────
    if success:
        try:
            get_telemetry(session_id).increment("tasks_succeeded")
        except Exception as tel_err:
            logger.debug(f"[Executor] Telemetry increment failed: {tel_err}")

    # ── 11. Return state updates ──────────────────────────────────────────────
    # NOTE: AgentState reducers for list fields use operator.add - return only
    # NEW items (not the full accumulated list) so LangGraph appends correctly.
    return {
        "current_step":       step + 1,
        "full_code_outputs":  [full_output],
        "code_outputs":       [truncated_output],
        "visualizations":     visualizations,
        "saved_artifacts":    saved_artifacts,
        "errors":             errors_out,
        "current_step_files": files,
        "_last_files_created": files,
        "_last_success":      success,
        "files_created":      files,
    }


def _visualizer_node(state):
    """Generate visualizations using VisualizationAgent and attach tuning results."""
    import pandas as pd
    from pathlib import Path
    from multimodal_ds.agents.visualization_agent import VisualizationAgent

    try:
        tab_summaries = state.get("tabular_summaries", [])
        if not tab_summaries:
            logger.info("[Visualizer] No tabular summaries – skipping.")
            return {}
        first = tab_summaries[0]
        file_path = first.get("source")
        if not file_path:
            logger.info("[Visualizer] Tabular summary missing source – skipping.")
            return {}

        p = Path(file_path)
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        elif p.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(file_path)
        elif p.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            logger.warning(f"[Visualizer] Unsupported file extension for {file_path}")
            return {}

        target_col = first.get("automl_suggestion", {}).get("target_candidates", [None])[0]
        vis_agent = VisualizationAgent(session_id=state.get("session_id", "default"))
        tuning = state.get("tuning_results", {})
        if tuning:
            vis_agent.set_tuning_results(tuning)
        manifest = vis_agent.generate(df=df, target_col=target_col)
        viz_files = [c["filename"] for c in manifest.charts]
        logger.info(f"[Visualizer] Generated {len(viz_files)} visualization files")
        return {"visualizations": viz_files}
    except Exception as e:
        logger.warning(f"[Visualizer] Node error: {e}")
        return {}


def _reflection_node(state):
    """
    Consolidated reflection node: diagnoses failures using ReflectionAgent and
    updates analysis tasks for retry if within MAX_RETRIES limit.
    """
    session_id = state.get("session_id", "default")
    retry_count = state.get("retry_count", 0)
    current_step = state.get("current_step", 0)

    try:
        # 1. Guard - if we've hit MAX_RETRIES, stop and proceed to reporting
        if retry_count >= MAX_RETRIES:
            logger.warning(f"[Reflection] MAX_RETRIES ({MAX_RETRIES}) reached for session {session_id}. Stopping loop.")
            return {"_loop_exit": True}

        logger.info(f"[Reflection] Starting diagnosis. Retry attempt {retry_count + 1}/{MAX_RETRIES}")

        # 2. Check verdict - if PASS, we don't need to reflect or retry
        eval_report_raw = state.get("eval_report", {})
        if not isinstance(eval_report_raw, dict):
            eval_report_dict = eval_report_raw.to_dict() if hasattr(eval_report_raw, "to_dict") else {}
        else:
            eval_report_dict = eval_report_raw

        if eval_report_dict.get("session_verdict") == "PASS":
            logger.info("[Reflection] Verdict is PASS. Skipping reflection.")
            return {}

        # 3. Call the real ReflectionAgent API
        agent = ReflectionAgent(session_id=session_id, max_retries=MAX_RETRIES)
        report = agent.reflect(eval_report=eval_report_dict, state=state)
        reflection_result = report.to_dict()

        # 4. Persist to SharedContextPool for UI/Audit
        pool = get_context_pool(session_id)
        pool.set("last_reflection", reflection_result, agent="reflection")

        # 5. Handle Retry Logic (Rewriting tasks and rewinding step)
        tasks = list(state.get("analysis_tasks", []))
        improved = report.improved_instructions
        
        # Rewind current_step by 1 so executor reruns the failed task
        retry_step = max(0, current_step - 1)
        
        if tasks and retry_step < len(tasks):
            # Augment the failed task's description with improved instructions
            orig_desc = tasks[retry_step].get("description", "")
            tasks[retry_step] = {
                **tasks[retry_step],
                "description": f"{orig_desc}\n\n[REFLECTION RETRY GUIDANCE]: {improved}",
            }
            logger.info(f"[Reflection] Rewrote task '{tasks[retry_step].get('name')}' for retry.")

        # 6. Return updated state
        return {
            "retry_count": retry_count + 1,
            "analysis_tasks": tasks,
            "current_step": retry_step,
            "reflection_report": reflection_result,
            "reflections": state.get("reflections", []) + [reflection_result],
            "errors": [],  # Clear errors for the retry attempt
            "_last_success": False, # Reset success flag
            "target_agent": report.target_agent,
        }

    except Exception as e:
        logger.error(f"[Reflection] Node failed: {e}")
        # On failure, at least increment counter to avoid infinite loops if the edge logic depends on it
        return {"retry_count": retry_count + 1}


def _reviewer_node(state):
    tasks   = state.get("analysis_tasks", [])
    outputs = state.get("full_code_outputs", [])
    errors  = state.get("errors", [])
    vizs    = state.get("visualizations", [])
    arts    = state.get("saved_artifacts", [])

    # Build all files created across session
    all_files = list(vizs) + list(arts)

    task_results = []
    for i, (task, output) in enumerate(zip(tasks, outputs)):
        step_num = i + 1
        task_failed = any(f"Step {step_num}:" in e for e in errors)
        
        # Determine files relevant to this step safely – log any issues but continue
        step_files = []
        try:
            for fname in all_files:
                if fname.lower().endswith(('.png', '.jpg', '.csv', '.pkl', '.joblib', '.html', '.txt')):
                    step_files.append(fname)
            # Include per-step file list if available
            files_per_step = state.get("_files_per_step", [])
            if i < len(files_per_step):
                step_files.extend(files_per_step[i])
        except Exception as e:
            logger.warning(f"[Reviewer] File aggregation failed: {e}")
        # Also scan output text for saved file references
        try:
            import re
            file_refs = re.findall(r'[\w\-]+\.\w{2,5}', output)
            known_exts = {'.png', '.jpg', '.csv', '.pkl', '.joblib', '.html', '.txt', '.json', '.parquet'}
            for ref in file_refs:
                if any(ref.endswith(ext) for ext in known_exts) and ref not in step_files:
                    step_files.append(ref)
        except Exception as e:
            logger.warning(f"[Reviewer] Output file reference parsing failed: {e}")


        task_results.append({
            "name":           task.get("name", f"step_{step_num}"),
            "success":        not task_failed,
            "output_preview": output,
            "files_created":  step_files,
            "error":          "",
        })

    session_id = state.get("session_id", "default")
    data_context = _build_data_context_for_eval(state)
    try:
        eval_agent = EvaluationAgent(session_id=session_id)
        stat_report = state.get("statistical_report", {})
        report = eval_agent.evaluate_task_results(
            task_results=task_results,
            data_context=data_context,
            stat_report=stat_report if stat_report else None,
        )
        return {"eval_report": report.to_dict()}
    except Exception as e:
        logger.warning(f"[Reviewer] EvaluationAgent failed: {e} - returning empty report")
        return {"eval_report": {
            "session_id": session_id,
            "task_count": len(task_results),
            "flagged_count": 0,
            "pass_count": len(task_results),
            "overall_session_score": 5.0,
            "session_verdict": "UNKNOWN",
            "evaluations": [],
        }}


def _build_data_context_for_eval(state: dict) -> str:
    """Build rich data context string for the evaluation agent."""
    parts = []
    for t in state.get("tabular_summaries", [])[:2]:
        cols = t.get("columns", [])
        shape = t.get("shape", [])
        parts.append(f"Dataset: {shape[0] if shape else '?'} rows × {shape[1] if len(shape) > 1 else '?'} cols")
        parts.append(f"Columns: {', '.join(str(c) for c in cols[:20])}")
        profile = t.get("data_profile", {})
        if profile.get("numeric_stats"):
            stats_preview = list(profile["numeric_stats"].items())[:3]
            for col, s in stats_preview:
                parts.append(f"  {col}: mean={s.get('mean', 0):.2f}, std={s.get('std', 0):.2f}")
    return "\n".join(parts)




def _reporter_node(state):
    from multimodal_ds.agents.reporter import reporter_agent
    return reporter_agent(state)

# ── Quality gate node ────────────────────────────────────────────────────────

def _quality_gate_node(state):
    """Fast rule‑based quality gate between executor and reviewer."""
    # Last execution info
    outputs = state.get("full_code_outputs", [])
    last_output = outputs[-1] if outputs else ""
    last_files = state.get("_last_files_created", [])
    last_success = state.get("_last_success", False)
    current_step = state.get("current_step", 0)
    tasks = state.get("analysis_tasks", [])
    task_name = "unknown"
    if current_step > 0 and tasks:
        task_name = tasks[current_step - 1].get("name", "unknown")
    # Rule checks
    gate_passed = True
    gate_reasons = []
    # 1. OS‑level success
    if not last_success:
        gate_passed = False
        gate_reasons.append("Execution returned non-zero exit code")
    # 2. Output length
    if len(last_output.strip()) < 20:
        gate_passed = False
        gate_reasons.append("Output too short - likely crashed silently")
    # 3. Expected artifact files for modeling/evaluation
    task_type = tasks[current_step - 1].get("type", "") if current_step > 0 and tasks else ""
    if task_type in ("modeling", "evaluation"):
        if not any(f.endswith((".pkl", ".joblib", ".csv", ".txt")) for f in last_files):
            gate_passed = False
            gate_reasons.append(f"Modeling task produced no artifact files: {last_files}")
    # 4. Python error indicators
    error_indicators = ["Traceback (most recent", "Error:", "Exception:", "ModuleNotFoundError", "KeyError:", "AttributeError:"]
    if any(ind in last_output for ind in error_indicators) and not last_files:
        gate_passed = False
        gate_reasons.append("Output contains Python errors with no recovery files")
    # 5. Hallucination guard – column names
    if state.get("tabular_summaries"):
        actual_cols = set(state["tabular_summaries"][0].get("columns", []))
        import re
        matches = re.findall(r"df\[\'([^\']+)\'\]|df\[\"([^\"]+)\"\]", last_output)
        mentioned_cols = {c for pair in matches for c in pair if c}
        phantom_cols = mentioned_cols - actual_cols
        if phantom_cols and len(phantom_cols) > 2:
            gate_passed = False
            gate_reasons.append(f"Hallucinated column names: {phantom_cols}")
    # Logging
    if gate_passed:
        logger.info(f"[QualityGate] Step {current_step} '{task_name}': PASSED")
    else:
        logger.warning(f"[QualityGate] Step {current_step} '{task_name}': FAILED - {gate_reasons}")
    # Update errors list
    new_errors = state.get("errors", [])
    if not gate_passed:
        new_errors = new_errors + [f"Quality gate failed: {r}" for r in gate_reasons]
    return {
        "gate_passed": gate_passed,
        "gate_reasons": gate_reasons,
        "errors": new_errors,
    }


# ── Conditional edges ────────────────────────────────────────────────────────

def _multi_ingest_router_node(state):
    """Concurrent ingestion of multiple file types.
    Reads routing flags from state['_routing_flags'] and invokes the relevant
    ingestion node functions in parallel using a ThreadPoolExecutor.
    Returns a merged state dict where list values are concatenated.
    """
    import concurrent.futures
    flags = state.get("_routing_flags", {})
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        if flags.get("table"):
            futures.append(executor.submit(_tab_ingest_node, state))
        if flags.get("doc"):
            futures.append(executor.submit(_doc_ingest_node, state))
        if flags.get("image"):
            futures.append(executor.submit(_img_ingest_node, state))
        if flags.get("audio"):
            futures.append(executor.submit(_audio_ingest_node, state))
        # Wait for all futures with timeout
        concurrent.futures.wait(futures, timeout=300)
    merged = {}
    for fut in futures:
        try:
            result = fut.result()
        except Exception as e:
            logger.warning(f"[MultiIngest] Ingestion future failed: {e}")
            continue
        for k, v in result.items():
            if isinstance(v, list) and isinstance(merged.get(k), list):
                merged[k] = merged[k] + v
            else:
                merged[k] = v
    return merged

def _decide_after_gate(state: dict) -> str:
    """Decide next step after quality gate.
    Returns "executor", "reflection", or "reporter".
    """
    gate_passed = state.get("gate_passed", True)
    current_step = state.get("current_step", 0)
    steps_total = state.get("steps_total", 0)
    retry_count = state.get("retry_count", 0)

    # 1. Happy path: Success and more tasks remain
    if gate_passed and current_step < steps_total:
        return "executor"
    
    # 2. Retry path: Failed gate and retries remain
    if not gate_passed and retry_count < MAX_RETRIES:
        return "reflection"
    
    # 3. Success and all tasks done -> Proceed to final review
    if gate_passed and current_step >= steps_total:
        return "reviewer"
        
    # 4. Terminal or Finish path
    return "reporter"

def _decide_reflection_outcome(state: dict) -> str:
    """Decide next step after reflection based on targeting.
    
    If _loop_exit is set (max retries reached), proceed to visualizer/reporter.
    Otherwise, route to the target agent defined by the ReflectionAgent.
    """
    if state.get("_loop_exit"):
        return "reporter"
        
    return state.get("target_agent", "executor")

def _decide_review_outcome(state: dict) -> str:
    """
    Decide the next step after the Reviewer agent evaluates task outputs.

    Logic:
    1. If more tasks remain -> return 'executor'
    2. If reflection needed (failed tasks/low score) and retries remain -> return 'reflection'
    3. If retries exhausted but still failing -> return 'reporter'
    4. Otherwise -> return 'visualizer' (to summarize results before reporting)
    """
    eval_report = state.get("eval_report", {})
    # Handle object vs dict
    if not isinstance(eval_report, dict) and hasattr(eval_report, "to_dict"):
        eval_report_dict = eval_report.to_dict()
    else:
        eval_report_dict = eval_report

    current_step = state.get("current_step", 0)
    steps_total = state.get("steps_total", 0)
    retry_count = state.get("retry_count", 0)

    # 1. More tasks remain
    if current_step < steps_total:
        return "executor"

    # 2. Reflection needed?
    flagged_count = eval_report_dict.get("flagged_count", 0)
    overall_score = eval_report_dict.get("overall_session_score", 10.0)
    reflection_needed = (flagged_count > 0 or overall_score < FLAG_OVERALL_THRESHOLD)

    if reflection_needed and retry_count < MAX_RETRIES:
        return "reflection"

    # NEW — retries exhausted but still failing: go to reporter directly
    if reflection_needed and retry_count >= MAX_RETRIES:
        logger.warning(
            f"[ReviewOutcome] Max retries exhausted with flagged_count={flagged_count} "
            f"score={overall_score:.2f} — routing to reporter (fail-safe)"
        )
        return "reporter"

    # 3. Clean pass — proceed to visualization
    return "visualizer"


# ── Graph builder ────────────────────────────────────────────────────────────

def build_graph(use_sqlite_checkpointer: bool = False, sqlite_path: str = "./checkpoints.db"):
    # pyrefly: ignore [missing-import]
    from langgraph.graph import StateGraph, END
    from multimodal_ds.core.state import AgentState

    builder = StateGraph(AgentState)

    builder.add_node("problem_understanding", _problem_understanding_node)
    builder.add_node("router", _router_node)
    # builder.add_node("decide_ingestion_path", _decide_ingestion_path) # No longer a node
    builder.add_node("doc_ingest", _doc_ingest_node)
    builder.add_node("img_ingest", _img_ingest_node)
    builder.add_node("audio_ingest", _audio_ingest_node)
    builder.add_node("tab_ingest", _tab_ingest_node)
    builder.add_node("merge_ingest", _ingest_merge_node)
    # builder.add_node("multi_ingest", _multi_ingest_router_node)
    builder.add_node("stats_val", _stats_validation_node)
    builder.add_node("model_selection", _model_selection_node)
    builder.add_node("planner", _planner_node)
    builder.add_node("visualizer", _visualizer_node)
    builder.add_node("quality_gate", _quality_gate_node)
    builder.add_node("reviewer", _reviewer_node)
    builder.add_node("reflection", _reflection_node)
    builder.add_node("executor", _executor_node)
    builder.add_node("reporter", _reporter_node)

    builder.set_entry_point("problem_understanding")

    builder.add_edge("problem_understanding", "router")
    
    # Conditional fan-out for ingestion
    builder.add_conditional_edges(
        "router",
        _decide_ingestion_path,
        ["doc_ingest", "img_ingest", "audio_ingest", "tab_ingest", "merge_ingest"]
    )
    
    # Merge results from all ingestion nodes
    builder.add_edge("doc_ingest", "merge_ingest")
    builder.add_edge("img_ingest", "merge_ingest")
    builder.add_edge("audio_ingest", "merge_ingest")
    builder.add_edge("tab_ingest", "merge_ingest")
    
    # Continue with validation after merging
    builder.add_edge("merge_ingest", "stats_val")
    # builder.add_edge("router", "multi_ingest")
    # After multi_ingest, always go through stats validation (no‑op if no table)
    # builder.add_edge("multi_ingest", "stats_val")
    builder.add_edge("stats_val",       "model_selection")
    builder.add_edge("model_selection", "planner")
    builder.add_edge("planner",         "executor")

    # The Core Execution Loop with Quality Gate
    builder.add_edge("executor", "quality_gate")

    builder.add_conditional_edges(
        "quality_gate",
        _decide_after_gate,
        {
            "executor":   "executor",
            "reflection": "reflection",
            "reviewer":   "reviewer",
            "reporter":   "reporter"
        }
    )

    # Secondary Review/Reflection paths
    builder.add_conditional_edges(
        "reviewer",
        _decide_review_outcome,
        {
            "executor":   "executor",
            "reflection": "reflection",
            "visualizer": "visualizer",
            "reporter":   "reporter"
        }
    )

    builder.add_conditional_edges(
        "reflection",
        _decide_reflection_outcome,
        {
            "executor":   "executor",
            "planner":    "planner",
            "visualizer": "visualizer",
            "reporter":   "reporter"
        }
    )

    # Termination Path
    builder.add_edge("visualizer", "reporter")
    builder.add_edge("reporter",   END)


    if use_sqlite_checkpointer:
        try:
            # pyrefly: ignore [missing-import]
            from langgraph.checkpoint.sqlite import SqliteSaver
            memory = SqliteSaver.from_conn_string(sqlite_path)
        except ImportError:
            # pyrefly: ignore [missing-import]
            from langgraph.checkpoint.memory import MemorySaver
            memory = MemorySaver()
    else:
        # pyrefly: ignore [missing-import]
        from langgraph.checkpoint.memory import MemorySaver
        memory = MemorySaver()

    return builder.compile(checkpointer=memory)


def make_initial_state(
    user_query: str,
    uploaded_files: list[str],
    session_id: Optional[str] = None,
) -> dict:
    return {
        "user_query":         user_query,
        "uploaded_files":     uploaded_files,
        "_routing_flags":     {},
        "parsed_documents":   [],
        "image_embeddings":   [],
        "audio_transcripts":  [],
        "tabular_summaries":  [],
        "statistical_report": {},
        "model_selection": {},
        "tuning_results": {},
        "analysis_plan":      "",
        "analysis_tasks":     [],
        "hypotheses":         [],
        "current_step":       0,
        "steps_total":        0,
        "code_outputs":       [],
        "full_code_outputs": [],
        "visualizations":     [],
        "saved_artifacts":    [],
        "gate_passed":        True,
        "gate_reasons":      [],
        "reflections":        [],
        "vector_store_id":    "",
        "retrieved_context":  "",
        "eval_report":        {},
        "reflection_report":  {},
        "final_report":       "",
        "problem_spec":       {},
        "session_id":         session_id or str(uuid.uuid4())[:8],
        "messages":           [],
        "_last_task_name":    "",
        "_last_files_created": [],
        "current_step_files": [],
        "current_step_success": False,
        "_files_per_step": [],
        "executive_summary":        "",
        "business_recommendations": [],
        "step_failures": {},
    }
