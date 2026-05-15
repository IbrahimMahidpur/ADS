from multimodal_ds.cli import _merge_state
from multimodal_ds.graph import make_initial_state

def test_cli_stream_accumulates_code_outputs():
    updates = [
        {"code_outputs": ["step 1 output"]},
        {"code_outputs": ["step 2 output"]},
    ]
    state = make_initial_state("test", [])
    for u in updates:
        state = _merge_state(state, u)
    assert len(state["code_outputs"]) == 2
