"""Count tokenizer ids used by VLA/VLM training text metadata.

The main use case is checking how many *different tokenizer ids* appear in
robot task/instruction strings, without downloading video frames or model
weights.

Examples:
    uv run python scripts/count_training_tokens.py --preset libero
    uv run python scripts/count_training_tokens.py --preset tokenizer --checkpoint HuggingFaceVLA/smolvla_libero
    uv run python scripts/count_training_tokens.py --repo lerobot/libero --output results/token_vocab/libero.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.errors import EntryNotFoundError
from transformers import AutoProcessor, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

DEFAULT_CHECKPOINT = "HuggingFaceVLA/smolvla_libero"
DEFAULT_TOKENIZER = "HuggingFaceTB/SmolVLM2-500M-Instruct"
LIBERO_REPOS = {
    "libero": "lerobot/libero",
    "spatial": "lerobot/libero_spatial_image",
    "object": "lerobot/libero_object_image",
    "goal": "lerobot/libero_goal_image",
    "long": "lerobot/libero_10_image",
}
COMMUNITY_REPOS = [
    "HuggingFaceVLA/community_dataset_v1",
    "HuggingFaceVLA/community_dataset_v2",
]
TEXT_FIELDS = (
    "task",
    "task_name",
    "name",
    "instruction",
    "description",
    "single_task",
    "__index_level_0__",
)
TASK_INDEX_FIELDS = ("task_index", "task_id", "id", "index")


@dataclass
class TextRecord:
    text: str
    weight: int = 1
    source: str = ""
    task_index: int | None = None


@dataclass
class RepoStats:
    repo_id: str
    task_source: str | None = None
    episode_source: str | None = None
    task_count: int = 0
    unique_text_count: int = 0
    episode_count: int = 0
    frame_count: int = 0
    records: list[TextRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class TokenSummary:
    name: str
    tokenizer_name: str
    tokenizer_len: int
    tokenizer_vocab_size: int | None
    max_length: int
    append_newline: bool
    include_special_tokens: bool
    text_count: int
    unique_text_count: int
    weighted_text_count: int
    total_tokens_unweighted: int
    total_tokens_weighted: int
    unique_token_ids: int
    tokenizer_coverage_pct: float
    unique_words: int
    repos: list[dict[str, Any]]
    token_ids: list[int]
    token_strings: list[str]
    text_examples: list[str]
    warnings: list[str] = field(default_factory=list)


def _offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _hf_download(repo_id: str, filename: str, repo_type: str = "dataset") -> Path:
    return Path(hf_hub_download(repo_id, filename, repo_type=repo_type, local_files_only=_offline()))


def _first_existing(files: set[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in files:
            return candidate
    return None


def _load_parquet_rows(repo_id: str, data_files: list[str]) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - optional dependency error path
        raise RuntimeError("Reading parquet metadata requires the `datasets` package.") from exc

    ds = load_dataset(repo_id, data_files=data_files, split="train")
    return [dict(ds[i]) for i in range(len(ds))]


def _extract_text(row: dict[str, Any]) -> str | None:
    for key in TEXT_FIELDS:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()
    return None


def _extract_task_index(row: dict[str, Any]) -> int | None:
    for key in TASK_INDEX_FIELDS:
        value = row.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def _load_task_rows(repo_id: str, files: set[str], prefix: str = "") -> tuple[list[dict[str, Any]], str | None]:
    candidates = [
        f"{prefix}meta/tasks.jsonl",
        f"{prefix}meta/tasks.json",
        f"{prefix}meta/tasks",
        f"{prefix}meta/tasks.parquet",
    ]
    path = _first_existing(files, candidates)
    if path is None:
        return [], None
    if path.endswith(".jsonl"):
        return _load_jsonl(_hf_download(repo_id, path)), path
    if path.endswith(".parquet"):
        return _load_parquet_rows(repo_id, [path]), path

    loaded = _load_json(_hf_download(repo_id, path))
    if isinstance(loaded, list):
        return [r for r in loaded if isinstance(r, dict)], path
    if isinstance(loaded, dict):
        if isinstance(loaded.get("tasks"), list):
            return [r for r in loaded["tasks"] if isinstance(r, dict)], path
        rows = []
        for key, value in loaded.items():
            if isinstance(value, str):
                rows.append({"task_index": key, "task": value})
            elif isinstance(value, dict):
                item = dict(value)
                item.setdefault("task_index", key)
                rows.append(item)
        return rows, path
    return [], path


def _load_episode_rows(repo_id: str, files: set[str], prefix: str = "") -> tuple[list[dict[str, Any]], str | None]:
    jsonl = f"{prefix}meta/episodes.jsonl"
    if jsonl in files:
        return _load_jsonl(_hf_download(repo_id, jsonl)), jsonl

    parquet_files = sorted(
        p for p in files if p.startswith(f"{prefix}meta/episodes/") and p.endswith(".parquet")
    )
    if f"{prefix}meta/episodes.parquet" in files:
        parquet_files.insert(0, f"{prefix}meta/episodes.parquet")
    if parquet_files:
        return _load_parquet_rows(repo_id, parquet_files), f"{prefix}meta/episodes*.parquet"
    return [], None


def _repo_prefixes(files: set[str]) -> list[str]:
    prefixes = {""}
    for path in files:
        if path.endswith("meta/tasks.jsonl") or path.endswith("meta/tasks.parquet"):
            prefixes.add(path.rsplit("meta/tasks", 1)[0])
        elif path.endswith("meta/episodes.jsonl") or path.endswith("meta/episodes.parquet"):
            prefixes.add(path.rsplit("meta/episodes", 1)[0])
    return sorted(prefixes)


def load_lerobot_metadata_texts(repo_id: str, skip_episode_metadata: bool = False) -> RepoStats:
    """Load task/instruction texts from a LeRobot-format HF dataset repo."""

    repo_stats = RepoStats(repo_id=repo_id)
    files = set(list_repo_files(repo_id, repo_type="dataset"))
    prefixes = _repo_prefixes(files)
    if prefixes == [""]:
        prefixes = [""]

    records: list[TextRecord] = []
    task_sources: set[str] = set()
    episode_sources: set[str] = set()

    for prefix in prefixes:
        task_rows, task_source = _load_task_rows(repo_id, files, prefix)
        episode_rows: list[dict[str, Any]] = []
        episode_source: str | None = None
        if not skip_episode_metadata:
            episode_rows, episode_source = _load_episode_rows(repo_id, files, prefix)
        if task_source:
            task_sources.add(task_source)
        if episode_source:
            episode_sources.add(episode_source)

        task_map: dict[int, str] = {}
        for row in task_rows:
            text = _extract_text(row)
            task_index = _extract_task_index(row)
            if text is not None and task_index is not None:
                task_map[task_index] = text

        if not task_map:
            for row in episode_rows:
                text = _extract_text(row)
                task_index = _extract_task_index(row)
                if text is not None and task_index is not None:
                    task_map.setdefault(task_index, text)

        frame_weight_by_task: Counter[int] = Counter()
        episode_count_by_task: Counter[int] = Counter()
        for row in episode_rows:
            task_index = _extract_task_index(row)
            if task_index is None:
                continue
            episode_count_by_task[task_index] += 1
            try:
                length = int(row.get("length", 1))
            except Exception:
                length = 1
            frame_weight_by_task[task_index] += max(length, 1)

        for task_index, text in sorted(task_map.items()):
            weight = frame_weight_by_task.get(task_index, episode_count_by_task.get(task_index, 1))
            records.append(TextRecord(text=text, weight=int(weight), source=prefix.rstrip("/"), task_index=task_index))

        repo_stats.episode_count += len(episode_rows)
        repo_stats.frame_count += sum(int(row.get("length", 0) or 0) for row in episode_rows)

    repo_stats.records = records
    repo_stats.task_count = len(records)
    repo_stats.unique_text_count = len({r.text for r in records})
    repo_stats.task_source = ", ".join(sorted(task_sources)) or None
    repo_stats.episode_source = ", ".join(sorted(episode_sources)) or None
    if skip_episode_metadata:
        repo_stats.warnings.append("Skipped episode metadata; frame-weighted counts are not reported for this source.")
    if not records:
        repo_stats.warnings.append(
            "No task metadata found. For nested/private datasets, download locally and pass task strings via "
            "--text-file."
        )
    return repo_stats


def load_text_file(path: Path, weight: int = 1) -> RepoStats:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            records.append(TextRecord(text=text, weight=weight, source=str(path)))
    return RepoStats(
        repo_id=str(path),
        task_source=str(path),
        task_count=len(records),
        unique_text_count=len({r.text for r in records}),
        records=records,
    )


def resolve_tokenizer_name(checkpoint: str | None, tokenizer_name: str | None) -> tuple[str, int]:
    if tokenizer_name:
        return tokenizer_name, 48
    if checkpoint:
        if Path(checkpoint).is_dir():
            config_path = Path(checkpoint) / "config.json"
        else:
            config_path = Path(
                hf_hub_download(checkpoint, "config.json", local_files_only=_offline())
            )
        config = _load_json(config_path)
        return config.get("vlm_model_name", DEFAULT_TOKENIZER), int(config.get("tokenizer_max_length", 48))
    return DEFAULT_TOKENIZER, 48


def load_tokenizer(tokenizer_name: str) -> PreTrainedTokenizerBase:
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=_offline())
    except Exception as exc:
        processor = AutoProcessor.from_pretrained(tokenizer_name, local_files_only=_offline())
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(f"No tokenizer found in processor {tokenizer_name!r}") from exc
        return tokenizer


def normalize_for_policy(text: str, append_newline: bool) -> str:
    if append_newline and not text.endswith("\n"):
        return text + "\n"
    return text


def count_tokens(
    *,
    name: str,
    tokenizer_name: str,
    tokenizer: PreTrainedTokenizerBase,
    repo_stats: list[RepoStats],
    max_length: int,
    append_newline: bool,
    include_special_tokens: bool,
) -> TokenSummary:
    token_ids: set[int] = set()
    token_ids_unweighted: list[int] = []
    total_tokens_unweighted = 0
    total_tokens_weighted = 0
    words: set[str] = set()
    text_examples: list[str] = []
    warnings: list[str] = []

    all_records = [record for stats in repo_stats for record in stats.records]
    for stats in repo_stats:
        warnings.extend(stats.warnings)

    for record in all_records:
        text = normalize_for_policy(record.text, append_newline)
        encoded = tokenizer(
            text,
            add_special_tokens=include_special_tokens,
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        ids = [int(i) for i in encoded["input_ids"]]
        token_ids.update(ids)
        token_ids_unweighted.extend(ids)
        total_tokens_unweighted += len(ids)
        total_tokens_weighted += len(ids) * max(1, int(record.weight))
        words.update(w.lower() for w in re.findall(r"[A-Za-z0-9_']+", record.text))
        if len(text_examples) < 10 and record.text not in text_examples:
            text_examples.append(record.text)

    sorted_ids = sorted(token_ids)
    tokenizer_len = len(tokenizer)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    return TokenSummary(
        name=name,
        tokenizer_name=tokenizer_name,
        tokenizer_len=tokenizer_len,
        tokenizer_vocab_size=int(vocab_size) if vocab_size is not None else None,
        max_length=max_length,
        append_newline=append_newline,
        include_special_tokens=include_special_tokens,
        text_count=len(all_records),
        unique_text_count=len({r.text for r in all_records}),
        weighted_text_count=sum(max(1, int(r.weight)) for r in all_records),
        total_tokens_unweighted=total_tokens_unweighted,
        total_tokens_weighted=total_tokens_weighted,
        unique_token_ids=len(sorted_ids),
        tokenizer_coverage_pct=(len(sorted_ids) / tokenizer_len * 100.0) if tokenizer_len else 0.0,
        unique_words=len(words),
        repos=[_repo_stats_for_json(stats) for stats in repo_stats],
        token_ids=sorted_ids,
        token_strings=tokenizer.convert_ids_to_tokens(sorted_ids),
        text_examples=text_examples,
        warnings=warnings,
    )


def _repo_stats_for_json(stats: RepoStats) -> dict[str, Any]:
    data = asdict(stats)
    data.pop("records", None)
    return data


def print_summary(summary: TokenSummary) -> None:
    print(f"\n{summary.name}")
    print("=" * len(summary.name))
    print(f"tokenizer: {summary.tokenizer_name}")
    print(f"tokenizer length: {summary.tokenizer_len:,}")
    if summary.tokenizer_vocab_size is not None:
        print(f"tokenizer vocab_size: {summary.tokenizer_vocab_size:,}")
    print(f"unique instruction strings: {summary.unique_text_count:,}")
    print(f"instruction records: {summary.text_count:,}")
    print(f"weighted frame/example count: {summary.weighted_text_count:,}")
    print(f"unique token ids: {summary.unique_token_ids:,} ({summary.tokenizer_coverage_pct:.3f}% of tokenizer)")
    print(f"unique word-like strings: {summary.unique_words:,}")
    print(f"total instruction tokens: {summary.total_tokens_unweighted:,}")
    print(f"weighted instruction tokens: {summary.total_tokens_weighted:,}")
    if summary.token_strings:
        preview = ", ".join(summary.token_strings[:40])
        print(f"token preview: {preview}")
    if summary.warnings:
        print("warnings:")
        for warning in summary.warnings:
            print(f"  - {warning}")


def build_repo_list(args: argparse.Namespace) -> list[str]:
    repos: list[str] = []
    if args.preset == "libero":
        repos.extend(LIBERO_REPOS[key] for key in ("spatial", "object", "goal", "long"))
    elif args.preset == "libero_aggregate":
        repos.append(LIBERO_REPOS["libero"])
    elif args.preset == "community":
        repos.extend(COMMUNITY_REPOS)
    repos.extend(args.repo or [])
    return list(dict.fromkeys(repos))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=["tokenizer", "libero", "libero_aggregate", "community", "custom"],
        default="libero",
        help="Built-in source set. `tokenizer` only reports tokenizer capacity unless --repo/--text-file is given.",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint config used to resolve tokenizer.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer/processor id. Overrides --checkpoint config.")
    parser.add_argument("--repo", action="append", help="HF dataset repo id to inspect for LeRobot task metadata.")
    parser.add_argument(
        "--text-file",
        action="append",
        type=Path,
        help="Plain text file with one instruction per line.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--max-length", type=int, help="Tokenizer max length. Defaults to checkpoint config or 48.")
    parser.add_argument(
        "--no-append-newline",
        action="store_true",
        help="Do not append the newline that SmolVLA's processor adds to task strings.",
    )
    parser.add_argument(
        "--include-special-tokens",
        action="store_true",
        help="Include tokenizer special tokens. Default counts content tokens only.",
    )
    parser.add_argument(
        "--skip-episodes",
        action="store_true",
        help="Only read task metadata. Useful for large nested community repos.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parse_args()
    tokenizer_name, config_max_length = resolve_tokenizer_name(args.checkpoint, args.tokenizer)
    max_length = args.max_length or config_max_length
    tokenizer = load_tokenizer(tokenizer_name)

    repo_stats: list[RepoStats] = []
    skip_episodes = args.skip_episodes or args.preset == "community"
    for repo_id in build_repo_list(args):
        try:
            repo_stats.append(load_lerobot_metadata_texts(repo_id, skip_episode_metadata=skip_episodes))
        except EntryNotFoundError as exc:
            repo_stats.append(RepoStats(repo_id=repo_id, warnings=[f"Missing expected metadata: {exc}"]))
    for path in args.text_file or []:
        repo_stats.append(load_text_file(path))

    if not repo_stats:
        repo_stats = [RepoStats(repo_id="tokenizer_only")]

    summary = count_tokens(
        name=args.preset,
        tokenizer_name=tokenizer_name,
        tokenizer=tokenizer,
        repo_stats=repo_stats,
        max_length=max_length,
        append_newline=not args.no_append_newline,
        include_special_tokens=args.include_special_tokens,
    )
    print_summary(summary)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
