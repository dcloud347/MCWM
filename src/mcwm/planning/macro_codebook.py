"""Deterministic two-tick macro-action codebook construction.

The codebook is fitted only from VPT contractor episodes in the training split.
Discrete action templates are grouped exactly; four-dimensional two-tick camera
trajectories are then standardized and clustered within each group.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from mcwm.actions.schema import (
    INTERACTION_NAMES,
    MOVEMENT_NAMES,
    ActionSource,
    CanonicalActionTick,
)
from mcwm.data.episode_store import EpisodeStore
from mcwm.data.manifest import DatasetManifest


CODEBOOK_SCHEMA_VERSION = 1
BASIC_CODE_NAMES = (
    "noop",
    "forward",
    "turn_left",
    "turn_right",
    "jump",
    "attack",
    "use",
)


BoolTicks = Tuple[Tuple[bool, ...], Tuple[bool, ...]]
FloatPair = Tuple[float, float]
FloatTicks = Tuple[FloatPair, FloatPair]


@dataclass(frozen=True)
class MacroCodebookFitConfig:
    """Configuration whose complete value is recorded in the artifact."""

    macro_length: int = 2
    stride: int = 1
    max_clusters_per_group: int = 4
    min_group_samples: int = 32
    min_cluster_samples: int = 8
    max_codes: int = 256
    max_camera_degrees: float = 30.0
    camera_std_floor: float = 0.25
    residual_quantile: float = 0.99
    minimum_residual_limit: float = 3.0
    kmeans_iterations: int = 40
    max_samples_per_group: int = 20000
    max_tick_gap_ms: int = 300
    max_source_tick_gap_ms: int = 100
    model_fps: int = 4
    environment_fps: int = 20
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.macro_length != 2:
            raise ValueError("the first codebook version requires macro_length=2")
        integers = (
            self.stride,
            self.max_clusters_per_group,
            self.min_group_samples,
            self.min_cluster_samples,
            self.max_codes,
            self.kmeans_iterations,
            self.max_samples_per_group,
            self.max_tick_gap_ms,
            self.max_source_tick_gap_ms,
            self.model_fps,
            self.environment_fps,
        )
        if min(integers) <= 0:
            raise ValueError("codebook integer parameters must be positive")
        if self.max_codes < len(BASIC_CODE_NAMES):
            raise ValueError("max_codes must leave room for all basic codes")
        if (
            self.max_camera_degrees <= 0
            or self.camera_std_floor <= 0
            or self.minimum_residual_limit <= 0
        ):
            raise ValueError("camera limits must be positive")
        if not 0.0 < self.residual_quantile <= 1.0:
            raise ValueError("residual_quantile must be in (0, 1]")
        if 1000 % self.model_fps:
            raise ValueError("model_fps must divide 1000 ms exactly")
        if self.environment_fps % self.model_fps:
            raise ValueError("environment_fps must be divisible by model_fps")

    @property
    def model_tick_ms(self) -> int:
        return 1000 // self.model_fps

    @property
    def action_repeat(self) -> int:
        return self.environment_fps // self.model_fps


@dataclass(frozen=True)
class MacroLegalityMetadata:
    """Static legality facts; online state checks are applied by the planner."""

    v1_supported: bool
    requires_gui: bool
    cursor_present: bool
    reasons: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MacroLegalityMetadata":
        return cls(
            v1_supported=bool(data["v1_supported"]),
            requires_gui=bool(data["requires_gui"]),
            cursor_present=bool(data["cursor_present"]),
            reasons=tuple(str(value) for value in data.get("reasons", ())),
        )


@dataclass(frozen=True)
class MacroCode:
    """One discrete code and its canonical two-tick action trajectory."""

    code_id: int
    movement: BoolTicks
    interaction: BoolTicks
    hotbar: Tuple[int, int]
    gui_open: Tuple[bool, bool]
    camera_mean: FloatTicks
    camera_std: FloatTicks
    camera_residual_max: float
    gui_mode: str
    sample_count: int
    legality: MacroLegalityMetadata
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.movement) != 2 or any(
            len(tick) != len(MOVEMENT_NAMES) for tick in self.movement
        ):
            raise ValueError("movement must contain two canonical ticks")
        if len(self.interaction) != 2 or any(
            len(tick) != len(INTERACTION_NAMES) for tick in self.interaction
        ):
            raise ValueError("interaction must contain two canonical ticks")
        if len(self.hotbar) != 2 or any(not 0 <= value <= 9 for value in self.hotbar):
            raise ValueError("hotbar must contain two values in [0, 9]")
        if len(self.gui_open) != 2:
            raise ValueError("gui_open must contain two values")
        if self.gui_mode not in {"gameplay", "gui", "transition"}:
            raise ValueError("invalid gui_mode")
        if self.sample_count < 0 or self.camera_residual_max < 0:
            raise ValueError("sample counts and residual limits cannot be negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code_id": self.code_id,
            "name": self.name,
            "canonical": {
                "movement": [list(value) for value in self.movement],
                "interaction": [list(value) for value in self.interaction],
                "hotbar": list(self.hotbar),
                "gui_open": list(self.gui_open),
            },
            "camera_mean": [list(value) for value in self.camera_mean],
            "camera_std": [list(value) for value in self.camera_std],
            "camera_residual_max": self.camera_residual_max,
            "gui_mode": self.gui_mode,
            "sample_count": self.sample_count,
            "legality": self.legality.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MacroCode":
        canonical = data["canonical"]
        return cls(
            code_id=int(data["code_id"]),
            name=data.get("name"),
            movement=tuple(
                tuple(bool(item) for item in tick)
                for tick in canonical["movement"]
            ),  # type: ignore[arg-type]
            interaction=tuple(
                tuple(bool(item) for item in tick)
                for tick in canonical["interaction"]
            ),  # type: ignore[arg-type]
            hotbar=tuple(int(item) for item in canonical["hotbar"]),  # type: ignore[arg-type]
            gui_open=tuple(bool(item) for item in canonical["gui_open"]),  # type: ignore[arg-type]
            camera_mean=tuple(
                tuple(float(item) for item in tick)
                for tick in data["camera_mean"]
            ),  # type: ignore[arg-type]
            camera_std=tuple(
                tuple(float(item) for item in tick)
                for tick in data["camera_std"]
            ),  # type: ignore[arg-type]
            camera_residual_max=float(data["camera_residual_max"]),
            gui_mode=str(data["gui_mode"]),
            sample_count=int(data["sample_count"]),
            legality=MacroLegalityMetadata.from_dict(data["legality"]),
        )


@dataclass(frozen=True)
class MacroCodebook:
    """Serializable fixed action vocabulary used by CEM."""

    manifest_hash: str
    fit_config: MacroCodebookFitConfig
    provenance: Mapping[str, Any]
    codes: Tuple[MacroCode, ...]
    schema_version: int = CODEBOOK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CODEBOOK_SCHEMA_VERSION:
            raise ValueError(f"unsupported codebook schema: {self.schema_version}")
        if not self.manifest_hash:
            raise ValueError("manifest_hash must be non-empty")
        if not self.codes:
            raise ValueError("codebook must contain at least one code")
        if tuple(code.code_id for code in self.codes) != tuple(range(len(self.codes))):
            raise ValueError("code IDs must be contiguous and ordered")
        names = {code.name for code in self.codes if code.name is not None}
        missing = set(BASIC_CODE_NAMES) - names
        if missing:
            raise ValueError(f"codebook is missing basic codes: {sorted(missing)}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_hash": self.manifest_hash,
            "macro_length": self.fit_config.macro_length,
            "random_seed": self.fit_config.seed,
            "fit_config": asdict(self.fit_config),
            "provenance": dict(self.provenance),
            "codes": [code.to_dict() for code in self.codes],
        }

    @property
    def content_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)

    @classmethod
    def read(cls, path: Path) -> "MacroCodebook":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        config = MacroCodebookFitConfig(**data["fit_config"])
        return cls(
            manifest_hash=str(data["manifest_hash"]),
            fit_config=config,
            provenance=dict(data["provenance"]),
            codes=tuple(MacroCode.from_dict(value) for value in data["codes"]),
            schema_version=int(data.get("schema_version", CODEBOOK_SCHEMA_VERSION)),
        )


@dataclass
class _ActionGroup:
    template: Tuple[Any, ...]
    cameras: List[Tuple[float, float, float, float]]
    cursor_present: bool = False
    total_count: int = 0


def _gui_mode(gui_open: Sequence[bool]) -> str:
    if all(gui_open):
        return "gui"
    if any(gui_open):
        return "transition"
    return "gameplay"


def _template(block: Sequence[CanonicalActionTick]) -> Tuple[Any, ...]:
    return (
        tuple(tuple(action.movement) for action in block),
        tuple(tuple(action.interaction) for action in block),
        tuple(action.hotbar for action in block),
        tuple(action.gui_open for action in block),
    )


def _block_is_usable(
    block: Sequence[CanonicalActionTick], config: MacroCodebookFitConfig
) -> bool:
    if len(block) != config.macro_length:
        return False
    if any(not action.valid or action.source is not ActionSource.VPT for action in block):
        return False
    if any(
        abs(value) > config.max_camera_degrees
        for action in block
        for value in action.camera
    ):
        return False
    return all(
        0 < current.timestamp_ms - previous.timestamp_ms <= config.max_tick_gap_ms
        for previous, current in zip(block, block[1:])
    )


def resample_actions_to_model_ticks(
    actions: Sequence[CanonicalActionTick],
    config: MacroCodebookFitConfig = MacroCodebookFitConfig(),
) -> Tuple[CanonicalActionTick, ...]:
    """Aggregate source-rate VPT rows into canonical 4 FPS model ticks.

    Held buttons use majority state, interaction events use any-active, camera
    deltas are summed, and the last hotbar event/GUI state wins. Empty, damaged,
    non-VPT, or discontinuous source windows become invalid padding ticks and are
    consequently filtered before fitting.
    """

    if not actions:
        return ()
    ordered = tuple(sorted(actions, key=lambda action: action.timestamp_ms))
    start = ordered[0].timestamp_ms
    end = ordered[-1].timestamp_ms
    result = []
    source_index = 0
    for timestamp in range(start, end + 1, config.model_tick_ms):
        window_end = timestamp + config.model_tick_ms
        window = []
        while source_index < len(ordered) and ordered[source_index].timestamp_ms < window_end:
            if ordered[source_index].timestamp_ms >= timestamp:
                window.append(ordered[source_index])
            source_index += 1
        continuous = bool(window) and all(
            current.timestamp_ms - previous.timestamp_ms
            <= config.max_source_tick_gap_ms
            for previous, current in zip(window, window[1:])
        )
        usable = continuous and all(
            action.valid and action.source is ActionSource.VPT for action in window
        )
        if not usable:
            result.append(
                CanonicalActionTick.noop(timestamp, ActionSource.VPT, valid=False)
            )
            continue
        movement = tuple(
            sum(action.movement[index] for action in window) * 2 >= len(window)
            for index in range(len(MOVEMENT_NAMES))
        )
        interaction = tuple(
            any(action.interaction[index] for action in window)
            for index in range(len(INTERACTION_NAMES))
        )
        hotbar_events = [action.hotbar for action in window if action.hotbar]
        camera = tuple(
            max(
                -180.0,
                min(180.0, sum(action.camera[index] for action in window)),
            )
            for index in range(2)
        )
        cursor = next(
            (action.cursor for action in reversed(window) if action.cursor is not None),
            None,
        )
        result.append(
            CanonicalActionTick(
                movement=movement,
                interaction=interaction,
                hotbar=hotbar_events[-1] if hotbar_events else 0,
                camera=camera,  # type: ignore[arg-type]
                cursor=cursor,
                gui_open=window[-1].gui_open,
                valid=True,
                timestamp_ms=timestamp,
                source=ActionSource.VPT,
                label_confidence=min(action.label_confidence for action in window),
            )
        )
    return tuple(result)


def _group_seed(seed: int, template: Tuple[Any, ...]) -> int:
    payload = json.dumps(template, separators=(",", ":"))
    digest = sha256(f"{seed}:{payload}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _mean_std(
    values: Sequence[Tuple[float, ...]], floor: float
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    dimensions = len(values[0])
    means = tuple(
        sum(value[index] for value in values) / len(values)
        for index in range(dimensions)
    )
    stds = tuple(
        max(
            floor,
            math.sqrt(
                sum((value[index] - means[index]) ** 2 for value in values)
                / len(values)
            ),
        )
        for index in range(dimensions)
    )
    return means, stds


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def _deterministic_kmeans(
    values: Sequence[Tuple[float, ...]], clusters: int, iterations: int, seed: int
) -> Tuple[Tuple[int, ...], Tuple[Tuple[float, ...], ...]]:
    """Small dependency-free k-means with seeded farthest-point initialization."""

    if not 1 <= clusters <= len(values):
        raise ValueError("clusters must fit the input")
    generator = random.Random(seed)
    centroids = [tuple(values[generator.randrange(len(values))])]
    while len(centroids) < clusters:
        distances = [
            min(_squared_distance(value, centroid) for centroid in centroids)
            for value in values
        ]
        next_index = max(range(len(values)), key=lambda index: (distances[index], -index))
        centroids.append(tuple(values[next_index]))

    assignments: Tuple[int, ...] = ()
    for _ in range(iterations):
        updated_assignments = tuple(
            min(
                range(clusters),
                key=lambda index: (_squared_distance(value, centroids[index]), index),
            )
            for value in values
        )
        if updated_assignments == assignments:
            break
        assignments = updated_assignments
        next_centroids = []
        for cluster in range(clusters):
            members = [value for value, assigned in zip(values, assignments) if assigned == cluster]
            if not members:
                next_centroids.append(centroids[cluster])
                continue
            next_centroids.append(
                tuple(
                    sum(value[index] for value in members) / len(members)
                    for index in range(len(values[0]))
                )
            )
        centroids = next_centroids
    return assignments, tuple(centroids)


def _legality(
    template: Tuple[Any, ...], cursor_present: bool, max_camera: float, camera: FloatTicks
) -> MacroLegalityMetadata:
    movement, interaction, _, gui_open = template
    reasons = []
    for tick in movement:
        if tick[MOVEMENT_NAMES.index("forward")] and tick[MOVEMENT_NAMES.index("back")]:
            reasons.append("mutually_exclusive_forward_back")
        if tick[MOVEMENT_NAMES.index("left")] and tick[MOVEMENT_NAMES.index("right")]:
            reasons.append("mutually_exclusive_left_right")
    allowed_interactions = {"attack", "use"}
    for tick in interaction:
        for index, active in enumerate(tick):
            if active and INTERACTION_NAMES[index] not in allowed_interactions:
                reasons.append(f"unsupported_interaction:{INTERACTION_NAMES[index]}")
        if tick[INTERACTION_NAMES.index("attack")] and tick[INTERACTION_NAMES.index("use")]:
            reasons.append("mutually_exclusive_attack_use")
    requires_gui = any(gui_open)
    if requires_gui:
        reasons.append("gui_planning_disabled")
    if cursor_present:
        reasons.append("cursor_planning_unsupported")
    if any(abs(value) > max_camera for tick in camera for value in tick):
        reasons.append("camera_limit")
    unique_reasons = tuple(sorted(set(reasons)))
    return MacroLegalityMetadata(
        v1_supported=not unique_reasons,
        requires_gui=requires_gui,
        cursor_present=cursor_present,
        reasons=unique_reasons,
    )


def _cluster_group(
    group: _ActionGroup, config: MacroCodebookFitConfig
) -> List[MacroCode]:
    values = group.cameras
    means, scales = _mean_std(values, config.camera_std_floor)
    standardized = [
        tuple((value[index] - means[index]) / scales[index] for index in range(4))
        for value in values
    ]
    cluster_count = min(
        config.max_clusters_per_group,
        max(1, len(values) // config.min_cluster_samples),
    )
    assignments, _ = _deterministic_kmeans(
        standardized,
        cluster_count,
        config.kmeans_iterations,
        _group_seed(config.seed, group.template),
    )
    result = []
    for cluster in range(cluster_count):
        members = [value for value, assigned in zip(values, assignments) if assigned == cluster]
        if len(members) < config.min_cluster_samples and len(values) >= config.min_group_samples:
            continue
        camera_mean, camera_std = _mean_std(members, config.camera_std_floor)
        residuals = sorted(
            math.sqrt(
                sum(
                    ((value[index] - camera_mean[index]) / camera_std[index]) ** 2
                    for index in range(4)
                )
            )
            for value in members
        )
        residual_index = max(0, math.ceil(config.residual_quantile * len(residuals)) - 1)
        movement, interaction, hotbar, gui_open = group.template
        camera_ticks = (camera_mean[:2], camera_mean[2:])
        std_ticks = (camera_std[:2], camera_std[2:])
        result.append(
            MacroCode(
                code_id=-1,
                movement=movement,
                interaction=interaction,
                hotbar=hotbar,
                gui_open=gui_open,
                camera_mean=camera_ticks,
                camera_std=std_ticks,
                camera_residual_max=max(
                    config.minimum_residual_limit,
                    residuals[residual_index],
                ),
                gui_mode=_gui_mode(gui_open),
                sample_count=len(members),
                legality=_legality(
                    group.template,
                    group.cursor_present,
                    config.max_camera_degrees,
                    camera_ticks,
                ),
            )
        )
    return result


def _all_false(ticks: Sequence[Sequence[bool]]) -> bool:
    return not any(value for tick in ticks for value in tick)


def _basic_candidate_score(name: str, code: MacroCode) -> Optional[Tuple[float, int]]:
    if code.gui_mode != "gameplay" or any(code.hotbar):
        return None
    movement = code.movement
    interaction = code.interaction
    camera_norm = sum(value * value for tick in code.camera_mean for value in tick)
    if name in {"noop", "turn_left", "turn_right"}:
        if not _all_false(movement) or not _all_false(interaction):
            return None
        yaw = sum(tick[1] for tick in code.camera_mean)
        if name == "noop" and abs(yaw) <= 1.0 and camera_norm <= 2.0:
            return camera_norm, -code.sample_count
        if name == "turn_left" and yaw < -0.25:
            return -abs(yaw), -code.sample_count
        if name == "turn_right" and yaw > 0.25:
            return -abs(yaw), -code.sample_count
        return None
    expected_movement = name if name in {"forward", "jump"} else None
    expected_interaction = name if name in {"attack", "use"} else None
    active_movement = {
        MOVEMENT_NAMES[index]
        for tick in movement
        for index, active in enumerate(tick)
        if active
    }
    active_interaction = {
        INTERACTION_NAMES[index]
        for tick in interaction
        for index, active in enumerate(tick)
        if active
    }
    if expected_movement is not None:
        if active_movement != {expected_movement} or active_interaction:
            return None
    elif active_interaction != {expected_interaction} or active_movement:
        return None
    return camera_norm, -code.sample_count


def _builtin_code(name: str, config: MacroCodebookFitConfig) -> MacroCode:
    movement = [[False] * len(MOVEMENT_NAMES) for _ in range(2)]
    interaction = [[False] * len(INTERACTION_NAMES) for _ in range(2)]
    camera = [[0.0, 0.0], [0.0, 0.0]]
    if name == "forward":
        for tick in movement:
            tick[MOVEMENT_NAMES.index("forward")] = True
    elif name == "jump":
        movement[0][MOVEMENT_NAMES.index("jump")] = True
    elif name in {"attack", "use"}:
        interaction[0][INTERACTION_NAMES.index(name)] = True
    elif name == "turn_left":
        camera[0][1] = camera[1][1] = -5.0
    elif name == "turn_right":
        camera[0][1] = camera[1][1] = 5.0
    template = (
        tuple(tuple(tick) for tick in movement),
        tuple(tuple(tick) for tick in interaction),
        (0, 0),
        (False, False),
    )
    camera_ticks = tuple(tuple(tick) for tick in camera)
    return MacroCode(
        code_id=-1,
        name=name,
        movement=template[0],
        interaction=template[1],
        hotbar=template[2],
        gui_open=template[3],
        camera_mean=camera_ticks,  # type: ignore[arg-type]
        camera_std=((config.camera_std_floor, config.camera_std_floor),) * 2,
        camera_residual_max=3.0,
        gui_mode="gameplay",
        sample_count=0,
        legality=_legality(
            template,
            False,
            config.max_camera_degrees,
            camera_ticks,  # type: ignore[arg-type]
        ),
    )


def _select_and_number(
    fitted: Sequence[MacroCode], config: MacroCodebookFitConfig
) -> Tuple[MacroCode, ...]:
    remaining = list(fitted)
    selected = []
    for name in BASIC_CODE_NAMES:
        candidates = [
            (score, index, code)
            for index, code in enumerate(remaining)
            for score in [_basic_candidate_score(name, code)]
            if score is not None
        ]
        if candidates:
            _, index, code = min(candidates, key=lambda value: (value[0], value[1]))
            selected.append(replace(code, name=name))
            remaining.pop(index)
        else:
            selected.append(_builtin_code(name, config))

    remaining.sort(
        key=lambda code: (
            not code.legality.v1_supported,
            -code.sample_count,
            code.gui_mode,
            code.movement,
            code.interaction,
            code.hotbar,
            code.camera_mean,
        )
    )
    selected.extend(remaining[: config.max_codes - len(selected)])
    return tuple(replace(code, code_id=index) for index, code in enumerate(selected))


def fit_macro_codebook_from_episodes(
    episodes: Sequence[Tuple[str, Sequence[CanonicalActionTick]]],
    *,
    manifest_hash: str,
    config: MacroCodebookFitConfig = MacroCodebookFitConfig(),
    provenance: Optional[Mapping[str, Any]] = None,
) -> MacroCodebook:
    """Fit from VPT training episodes already sampled at ``config.model_fps``."""

    groups: Dict[Tuple[Any, ...], _ActionGroup] = {}
    usable_blocks = 0
    filtered_blocks = 0
    for episode_id, actions in sorted(episodes, key=lambda value: value[0]):
        for start in range(0, len(actions) - config.macro_length + 1, config.stride):
            block = actions[start : start + config.macro_length]
            if not _block_is_usable(block, config):
                filtered_blocks += 1
                continue
            key = _template(block)
            group = groups.setdefault(key, _ActionGroup(key, []))
            group.total_count += 1
            group.cursor_present = group.cursor_present or any(
                action.cursor is not None for action in block
            )
            camera = tuple(value for action in block for value in action.camera)
            if len(group.cameras) < config.max_samples_per_group:
                group.cameras.append(camera)  # type: ignore[arg-type]
            else:
                # Stable per-group reservoir sampling keeps fitting bounded on large corpora.
                generator = random.Random(
                    _group_seed(config.seed + group.total_count, key)
                )
                replacement = generator.randrange(group.total_count)
                if replacement < config.max_samples_per_group:
                    group.cameras[replacement] = camera  # type: ignore[assignment]
            usable_blocks += 1

    fitted = []
    for key in sorted(groups, key=lambda value: json.dumps(value, separators=(",", ":"))):
        group = groups[key]
        # Rare groups are filtered unless they can supply one of the required basics.
        protects_basic = any(
            _basic_candidate_score(
                name,
                MacroCode(
                    code_id=-1,
                    movement=key[0],
                    interaction=key[1],
                    hotbar=key[2],
                    gui_open=key[3],
                    camera_mean=((0.0, 0.0), (0.0, 0.0)),
                    camera_std=((1.0, 1.0), (1.0, 1.0)),
                    camera_residual_max=0.0,
                    gui_mode=_gui_mode(key[3]),
                    sample_count=group.total_count,
                    legality=_legality(
                        key, group.cursor_present, config.max_camera_degrees,
                        ((0.0, 0.0), (0.0, 0.0)),
                    ),
                ),
            )
            is not None
            for name in BASIC_CODE_NAMES
        )
        if group.total_count < config.min_group_samples and not protects_basic:
            filtered_blocks += group.total_count
            continue
        fitted.extend(_cluster_group(group, config))

    details = dict(provenance or {})
    details.update(
        {
            "split": "train",
            "source": ActionSource.VPT.value,
            "episode_ids": [value[0] for value in sorted(episodes)],
            "episode_count": len(episodes),
            "usable_blocks": usable_blocks,
            "filtered_blocks": filtered_blocks,
            "group_count": len(groups),
        }
    )
    return MacroCodebook(
        manifest_hash=manifest_hash,
        fit_config=config,
        provenance=details,
        codes=_select_and_number(fitted, config),
    )


def build_macro_codebook(
    data_root: Path,
    *,
    config: MacroCodebookFitConfig = MacroCodebookFitConfig(),
) -> MacroCodebook:
    """Load exactly the VPT training split from an EpisodeStore and fit it."""

    root = Path(data_root)
    manifest = DatasetManifest.read(root / "dataset_manifest.json")
    selected = tuple(
        episode
        for episode in manifest.episodes
        if episode.split == "train" and episode.source is ActionSource.VPT
    )
    if not selected:
        raise ValueError("no VPT contractor episodes exist in the training split")
    store = EpisodeStore(root)
    episodes = tuple(
        (
            entry.episode_id,
            resample_actions_to_model_ticks(
                store.read_episode(entry.episode_id).actions,
                config,
            ),
        )
        for entry in sorted(selected, key=lambda value: value.episode_id)
    )
    return fit_macro_codebook_from_episodes(
        episodes,
        manifest_hash=manifest.content_hash,
        config=config,
        provenance={
            "dataset_root": str(root.resolve()),
            "action_sha256": {
                entry.episode_id: entry.action_sha256 for entry in selected
            },
        },
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("macro_codebook.json"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-codes", type=int, default=256)
    parser.add_argument("--min-group-samples", type=int, default=32)
    parser.add_argument("--min-cluster-samples", type=int, default=8)
    parser.add_argument("--max-clusters-per-group", type=int, default=4)
    args = parser.parse_args(argv)
    config = MacroCodebookFitConfig(
        seed=args.seed,
        max_codes=args.max_codes,
        min_group_samples=args.min_group_samples,
        min_cluster_samples=args.min_cluster_samples,
        max_clusters_per_group=args.max_clusters_per_group,
    )
    codebook = build_macro_codebook(args.data_root, config=config)
    codebook.write(args.output)
    print(f"wrote {len(codebook.codes)} codes to {args.output}")
    print(f"manifest_sha256={codebook.manifest_hash}")
    print(f"codebook_sha256={codebook.content_hash}")


if __name__ == "__main__":
    main()
