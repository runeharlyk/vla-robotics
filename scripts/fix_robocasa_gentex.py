from __future__ import annotations

from itertools import cycle, islice
from pathlib import Path
import yaml


FIXTURE_ASSET_DIRS = {
    "coffee_machine": ("fixtures/coffee_machines", "fixtures/coffee_machines"),
    "dishwasher": ("fixtures/dishwashers", "fixtures/dishwashers"),
    "hood": ("fixtures/hoods", "fixtures/hoods"),
    "microwave": ("fixtures/microwaves", "fixtures/microwaves"),
    "oven": ("fixtures/ovens", "fixtures/ovens"),
    "sink": ("fixtures/sinks", "fixtures/sinks"),
    "stove": ("fixtures/stoves", "fixtures/stoves"),
    "stovetop": ("fixtures/stovetops", "fixtures/stovetops"),
    "toaster": ("fixtures/toasters", "fixtures/toasters"),
    "window": ("fixtures/windows", "fixtures/windows"),
}

OBJECT_ASSET_DIRS = {
    "knife_block": ("objects/knife_block", "objects/knife_block"),
    "paper_towel": ("objects/paper_towel_holder", "objects/paper_towel_holder"),
    "plant": ("objects/plant", "objects/plant"),
    "stool": ("objects/stool", "objects/stool"),
    "utensil_rack": ("objects/utensil_rack", "objects/utensil_rack"),
}

STOOL_ALIAS_TARGETS = [
    "rattan_stool",
    "stool_1_1",
    "stool_1_2",
    "stool_1_3",
    "stool_2",
]

CABINET_PANEL_VARIANTS = [
    {"panel_type": "slab"},
    {"panel_type": "raised"},
    {"panel_type": "shaker"},
    {"panel_type": "divided_window"},
    {"panel_type": "full_window", "panel_config": "{opacity: 0.4}"},
    {"panel_type": "full_window", "panel_config": "{opacity: 0.1}"},
]

CABINET_HANDLE_VARIANTS = [
    {"handle_type": "bar", "handle_config": "{texture: textures/metals/bright_metal.png}"},
    {"handle_type": "boxed", "handle_config": "{texture: textures/metals/bright_metal.png}"},
    {"handle_type": "knob", "handle_config": "{texture: textures/metals/bright_metal.png}"},
    {"handle_type": "bar", "handle_config": "{texture: textures/metals/brass.png}"},
    {"handle_type": "bar", "handle_config": "{texture: textures/flat/black.png}"},
    {"handle_type": "bar", "handle_config": "{texture: textures/flat/dark_gray.png}"},
]


def _pngs(directory: Path, limit: int = 100) -> list[str]:
    return sorted(path.name for path in directory.glob("*.png"))[:limit]


def _append_lines(path: Path, lines: list[str], sentinel: str) -> None:
    text = path.read_text(encoding="utf-8")
    if sentinel in text:
        return

    suffix = "\n" if not text.endswith("\n") else ""
    path.write_text(text + suffix + "\n".join(lines) + "\n", encoding="utf-8")


def _simple_texture_aliases(prefix: str, rel_dir: str, filenames: list[str]) -> list[str]:
    lines: list[str] = []
    for idx, name in enumerate(filenames, start=1):
        lines.append(f"{prefix}{idx:03d}:")
        lines.append(f"  texture: {rel_dir}/{name}")
        lines.append("")
    return lines[:-1] if lines else lines


def _counter_aliases(counter_files: list[str], cabinet_files: list[str]) -> list[str]:
    lines: list[str] = []
    for idx, name in enumerate(counter_files, start=1):
        lines.append(f"top_gentex{idx:03d}:")
        lines.append(f"  top_texture: generative_textures/counter/{name}")
        lines.append("")
    for idx, name in enumerate(cabinet_files, start=1):
        lines.append(f"base_gentex{idx:03d}:")
        lines.append(f"  base_texture: generative_textures/cabinet/{name}")
        lines.append("")
    return lines[:-1] if lines else lines


def _asset_entries(scan_root: Path, xml_prefix: str) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    if not scan_root.exists():
        return entries

    for asset_dir in sorted(path for path in scan_root.iterdir() if path.is_dir()):
        if not (asset_dir / "model.xml").exists():
            continue
        entries[asset_dir.name] = {"xml": f"{xml_prefix}/{asset_dir.name}"}
    return entries


def _stool_entries() -> dict[str, dict[str, str]]:
    aliases = [f"Stool{idx:03d}" for idx in range(2, 26)]
    targets = list(islice(cycle(STOOL_ALIAS_TARGETS), len(aliases)))
    return {
        alias: {"xml": f"fixtures/accessories/stools/{target}"}
        for alias, target in zip(aliases, targets)
    }


def _cabinet_alias_entries() -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}

    for idx in range(1, 52):
        entries[f"CabinetDoorPanel{idx:03d}"] = dict(CABINET_PANEL_VARIANTS[(idx - 1) % len(CABINET_PANEL_VARIANTS)])
        entries[f"CabinetHandle{idx:03d}"] = dict(CABINET_HANDLE_VARIANTS[(idx - 1) % len(CABINET_HANDLE_VARIANTS)])

    return entries


def _append_registry_entries(path: Path, entries: dict[str, dict[str, object]]) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = {key: value for key, value in entries.items() if key not in data}
    if not missing:
        return

    lines: list[str] = []
    for key, values in missing.items():
        lines.append(f"{key}:")
        for field, value in values.items():
            lines.append(f"  {field}: {value}")
        lines.append("")
    _append_lines(path, lines[:-1], sentinel=f"{next(iter(missing))}:")


def _rewrite_missing_xml_roots(registry_dir: Path, assets_root: Path) -> None:
    """Fix stale `objects/lightwheel/...` registry entries to the extracted object root.

    The current NVIDIA asset bundle unpacks many object folders directly under
    `models/assets/objects/<category>/...`, while some upstream fixture registry
    entries still point at `objects/lightwheel/<category>/...`.
    """
    for registry_path in sorted(registry_dir.glob("*.yaml")):
        data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        changed = False
        for entry in data.values():
            if not isinstance(entry, dict):
                continue
            xml_rel = entry.get("xml")
            if not isinstance(xml_rel, str):
                continue
            if not xml_rel.startswith("objects/lightwheel/"):
                continue

            current_model = assets_root / xml_rel / "model.xml"
            if current_model.exists():
                continue

            candidate_rel = "objects/" + xml_rel[len("objects/lightwheel/") :]
            candidate_model = assets_root / candidate_rel / "model.xml"
            if candidate_model.exists():
                entry["xml"] = candidate_rel
                changed = True

        if changed:
            registry_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _sync_asset_registry(package_root: Path) -> None:
    assets_root = package_root / "models" / "assets"
    registry_dir = assets_root / "fixtures" / "fixture_registry"

    for registry_name, (scan_rel, xml_prefix) in FIXTURE_ASSET_DIRS.items():
        _append_registry_entries(
            registry_dir / f"{registry_name}.yaml",
            _asset_entries(assets_root / scan_rel, xml_prefix),
        )

    for registry_name, (scan_rel, xml_prefix) in OBJECT_ASSET_DIRS.items():
        _append_registry_entries(
            registry_dir / f"{registry_name}.yaml",
            _asset_entries(assets_root / scan_rel, xml_prefix),
        )

    stool_assets = _asset_entries(assets_root / "objects" / "stool", "objects/stool")
    if stool_assets:
        _append_registry_entries(registry_dir / "stool.yaml", stool_assets)
    else:
        _append_registry_entries(registry_dir / "stool.yaml", _stool_entries())

    _append_registry_entries(registry_dir / "cabinet.yaml", _cabinet_alias_entries())
    _rewrite_missing_xml_roots(registry_dir, assets_root)


def main() -> None:
    package_root = Path(".robocasa-src/robocasa").resolve()
    registry_dir = package_root / "models" / "assets" / "fixtures" / "fixture_registry"
    gentex_dir = package_root / "models" / "assets" / "generative_textures"

    wall_files = _pngs(gentex_dir / "wall")
    floor_files = _pngs(gentex_dir / "floor")
    cabinet_files = _pngs(gentex_dir / "cabinet")
    counter_files = _pngs(gentex_dir / "counter")

    if min(map(len, (wall_files, floor_files, cabinet_files, counter_files))) < 100:
        raise RuntimeError("Expected at least 100 generative texture PNGs in each RoboCasa texture directory.")

    _append_lines(
        registry_dir / "wall.yaml",
        _simple_texture_aliases("gentex", "generative_textures/wall", wall_files),
        sentinel="gentex001:",
    )
    _append_lines(
        registry_dir / "floor.yaml",
        _simple_texture_aliases("gentex", "generative_textures/floor", floor_files),
        sentinel="gentex001:",
    )
    _append_lines(
        registry_dir / "cabinet.yaml",
        _simple_texture_aliases("gentex", "generative_textures/cabinet", cabinet_files),
        sentinel="gentex001:",
    )
    _append_lines(
        registry_dir / "counter.yaml",
        _counter_aliases(counter_files, cabinet_files),
        sentinel="top_gentex001:",
    )
    _sync_asset_registry(package_root)

    print("Patched RoboCasa fixture registry with local gentex and asset aliases.")


if __name__ == "__main__":
    main()
