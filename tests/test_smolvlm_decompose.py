from __future__ import annotations

from typer.testing import CliRunner

from vla.__main__ import app
from vla.planning.smolvlm_decompose import (
    DecompositionResult,
    Subgoal,
    _extract_imperative_payload,
    _extract_json_payload,
    _extract_numbered_payload,
    _normalize_result,
    build_decomposition_prompt,
    resolve_vlm_model_id,
)


def test_extract_json_payload_handles_fenced_output() -> None:
    payload = _extract_json_payload(
        '```json\n{"task_summary":"move the bowl","subgoals":["reach bowl","place bowl"]}\n```'
    )

    assert payload["task_summary"] == "move the bowl"
    assert payload["subgoals"] == ["reach bowl", "place bowl"]


def test_normalize_result_accepts_string_subgoals() -> None:
    result = _normalize_result(
        {
            "task_summary": "Complete the long-horizon task.",
            "subgoals": ["open the drawer", "pick up the bowl", "place the bowl on the plate"],
        },
        instruction="put the bowl on the plate after opening the drawer",
        model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        raw_response="{}",
    )

    assert result.task_summary == "Complete the long-horizon task."
    assert [step.title for step in result.subgoals] == [
        "open the drawer",
        "pick up the bowl",
        "place the bowl on the plate",
    ]


def test_normalize_result_replaces_placeholder_summary() -> None:
    result = _normalize_result(
        {
            "task_summary": "<one sentence>",
            "subgoals": ["open the drawer", "pick up the bowl"],
        },
        instruction="open the drawer and pick up the bowl",
        model_id="stub-smolvlm",
        raw_response="Summary: <one sentence>",
    )

    assert result.task_summary == "open the drawer and pick up the bowl"


def test_extract_numbered_payload_handles_plain_text_plan() -> None:
    payload = _extract_numbered_payload(
        "Summary: Open the drawer, pick up the bowl, and place it on the plate.\n"
        "1. Open the top drawer.\n"
        "2. Pick up the black bowl.\n"
        "3. Place the black bowl on the plate.\n"
    )

    assert payload["task_summary"].startswith("Open the drawer")
    assert payload["subgoals"] == [
        "Open the top drawer.",
        "Pick up the black bowl.",
        "Place the black bowl on the plate.",
    ]


def test_extract_imperative_payload_handles_concatenated_actions() -> None:
    payload = _extract_imperative_payload("put the alphabet soup in the basket put the tomato sauce in the basket")

    assert payload["subgoals"] == [
        "put the alphabet soup in the basket",
        "put the tomato sauce in the basket",
    ]


def test_build_decomposition_prompt_mentions_constraints() -> None:
    prompt = build_decomposition_prompt(
        instruction="open the cabinet and move the bowl to the plate",
        benchmark_hint="LIBERO Long",
        scene_context="The bowl starts inside the cabinet.",
    )

    assert "LIBERO Long" in prompt
    assert "The bowl starts inside the cabinet." in prompt
    assert "Return only this format" in prompt
    assert "Summary:" in prompt
    assert "single-arm tabletop robot" in prompt


def test_resolve_vlm_model_id_reads_checkpoint_config(monkeypatch) -> None:
    monkeypatch.setattr(
        "vla.planning.smolvlm_decompose.SmolVLAPolicy._load_ckpt_config",
        lambda checkpoint: {"vlm_model_name": "stub-smolvlm"},
    )

    assert resolve_vlm_model_id(checkpoint="fake-checkpoint") == "stub-smolvlm"


def test_cli_decompose_formats_result(monkeypatch) -> None:
    runner = CliRunner()

    def fake_decompose_instruction(**_: object) -> DecompositionResult:
        return DecompositionResult(
            instruction="open the drawer and place the bowl on the plate",
            model_id="stub-smolvlm",
            task_summary="Open storage, retrieve the bowl, then place it.",
            subgoals=[
                Subgoal(1, "Open drawer", "Pull the drawer open enough to access the bowl.", "drawer is open"),
                Subgoal(2, "Place bowl", "Move the bowl from the drawer to the plate.", "bowl rests on plate"),
            ],
            raw_response="{}",
        )

    monkeypatch.setattr("vla.planning.smolvlm_decompose.decompose_instruction", fake_decompose_instruction)
    result = runner.invoke(app, ["decompose", "--instruction", "open the drawer and place the bowl on the plate"])

    assert result.exit_code == 0
    assert "Summary: Open storage, retrieve the bowl, then place it." in result.stdout
    assert "1. Open drawer" in result.stdout
