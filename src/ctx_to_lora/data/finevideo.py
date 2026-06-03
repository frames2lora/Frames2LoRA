import hashlib
import json
import math
import random
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ctx_to_lora.data.video_manifest import (
    DATA_ROOT,
    dump_jsonl,
    normalize_manifest_row,
    relativize_video_path,
)


SCENE_PROMPTS = (
    "Describe what is happening in this scene.",
    "Write a short summary of this scene.",
)

ADJACENT_PROMPTS = (
    "Summarize this video segment.",
    "Describe the sequence of events in this segment.",
)

FULL_PROMPTS = (
    "Summarize this video in 1-3 sentences.",
)

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SpanCandidate:
    sample_id: str
    split: str
    dataset: str
    span_type: str
    source_video_abs: str
    output_video_abs: str
    clip_start_sec: float | None
    clip_end_sec: float | None
    prompt: str
    target_text: str
    metadata: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _clean_text(text: str) -> str:
    cleaned = _WHITESPACE_RE.sub(" ", text.strip())
    cleaned = cleaned.strip(" \t\r\n")
    return cleaned


def _ensure_sentence(text: str) -> str:
    text = _clean_text(text)
    if not text:
        return ""
    if text[-1] not in {".", "!", "?"}:
        text += "."
    return text


def _choose_prompt(sample_id: str, *, span_type: str) -> str:
    if span_type == "scene":
        choices = SCENE_PROMPTS
    elif span_type == "adjacent":
        choices = ADJACENT_PROMPTS
    elif span_type == "full":
        choices = FULL_PROMPTS
    else:
        raise ValueError(f"Unknown span type: {span_type}")
    digest = hashlib.sha256(sample_id.encode()).digest()
    return choices[digest[0] % len(choices)]


def parse_hhmmss_to_seconds(value: str | None) -> float | None:
    if not value:
        return None
    value = str(value).strip()
    parts = value.split(":")
    if len(parts) != 3:
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    except ValueError:
        return None
    return hours * 3600.0 + minutes * 60.0 + seconds


def _scene_bounds(scene: dict[str, Any]) -> tuple[float | None, float | None]:
    timestamps = scene.get("timestamps") or {}
    if isinstance(timestamps, dict):
        start = parse_hhmmss_to_seconds(timestamps.get("start_timestamp"))
        end = parse_hhmmss_to_seconds(timestamps.get("end_timestamp"))
        if start is not None and end is not None and end > start:
            return start, end
    return None, None


def _scene_activity_descriptions(scene: dict[str, Any]) -> list[str]:
    activities = scene.get("activities") or []
    out: list[str] = []
    for activity in activities:
        if not isinstance(activity, dict):
            continue
        description = _ensure_sentence(str(activity.get("description") or ""))
        if description:
            out.append(description)
    return out


def _scene_target(scene: dict[str, Any]) -> str:
    activities = _scene_activity_descriptions(scene)
    if activities:
        return " ".join(activities[:2])
    scene_title = _ensure_sentence(str(scene.get("title") or ""))
    if scene_title:
        return scene_title
    mood = ((scene.get("mood") or {}).get("description") or "").strip()
    mood_text = _ensure_sentence(f"The scene mood is {mood.lower()}." if mood else "")
    return mood_text


def _adjacent_target(scenes: list[dict[str, Any]]) -> str:
    activities: list[str] = []
    for scene in scenes:
        activities.extend(_scene_activity_descriptions(scene))
    if activities:
        return " ".join(activities[:4])
    titles = [
        _ensure_sentence(str(scene.get("title") or ""))
        for scene in scenes
    ]
    titles = [title for title in titles if title]
    return " ".join(titles[:3])


def _full_target_from_description(meta: dict[str, Any]) -> str:
    description = _ensure_sentence(str((meta.get("description") or "")).strip())
    return description


def _span_stem(split: str, source_stem: str, span_type: str, span_key: str) -> str:
    return f"{split}-finevideo-{source_stem}-{span_type}-{span_key}"


def _clip_output_path(clips_root: Path, split: str, span_type: str, stem: str) -> Path:
    digest = hashlib.sha256(stem.encode()).hexdigest()[:6]
    return clips_root / split / span_type / digest / f"{stem}.mp4"


def _ffmpeg_extract_clip(
    source_video: Path,
    output_video: Path,
    start_sec: float,
    end_sec: float,
    overwrite: bool,
    *,
    ffmpeg_threads: int = 1,
    ffmpeg_preset: str = "ultrafast",
    ffmpeg_crf: int = 30,
    include_audio: bool = False,
) -> None:
    if output_video.exists() and not overwrite:
        return
    output_video.parent.mkdir(parents=True, exist_ok=True)
    duration = end_sec - start_sec
    if duration <= 0:
        raise ValueError(f"Invalid clip duration for {source_video}: {start_sec} -> {end_sec}")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        str(max(1, int(ffmpeg_threads))),
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(source_video),
        "-c:v",
        "libx264",
        "-preset",
        ffmpeg_preset,
        "-crf",
        str(int(ffmpeg_crf)),
        "-movflags",
        "+faststart",
    ]
    if include_audio:
        command.extend(["-c:a", "aac"])
    else:
        command.append("-an")
    command.extend(
        [
        "-y" if overwrite else "-n",
        str(output_video),
        ]
    )
    subprocess.run(command, check=True)


def _build_scene_candidate(
    *,
    split: str,
    source_video: Path,
    source_stem: str,
    scene: dict[str, Any],
    scene_index: int,
    clips_root: Path,
) -> SpanCandidate | None:
    start, end = _scene_bounds(scene)
    if start is None or end is None:
        return None
    target = _scene_target(scene)
    if not target:
        return None
    span_key = f"s{scene_index:02d}"
    stem = _span_stem(split, source_stem, "scene", span_key)
    output_path = _clip_output_path(clips_root, split, "scene", stem)
    prompt = _choose_prompt(stem, span_type="scene")
    metadata = {
        "dataset": "finevideo",
        "span_type": "scene",
        "source_video_path": relativize_video_path(str(source_video)),
        "scene_index": scene_index,
        "scene_id": scene.get("sceneId"),
        "scene_title": scene.get("title"),
    }
    return SpanCandidate(
        sample_id=stem,
        split=split,
        dataset="finevideo",
        span_type="scene",
        source_video_abs=str(source_video),
        output_video_abs=str(output_path),
        clip_start_sec=start,
        clip_end_sec=end,
        prompt=prompt,
        target_text=target,
        metadata=metadata,
    )


def _build_adjacent_candidate(
    *,
    split: str,
    source_video: Path,
    source_stem: str,
    scenes: list[dict[str, Any]],
    start_scene_index: int,
    window_size: int,
    clips_root: Path,
) -> SpanCandidate | None:
    window = scenes[start_scene_index : start_scene_index + window_size]
    if len(window) != window_size:
        return None
    first_start, _ = _scene_bounds(window[0])
    _, last_end = _scene_bounds(window[-1])
    if first_start is None or last_end is None or last_end <= first_start:
        return None
    target = _adjacent_target(window)
    if not target:
        return None
    span_key = f"a{start_scene_index:02d}w{window_size}"
    stem = _span_stem(split, source_stem, "adjacent", span_key)
    output_path = _clip_output_path(clips_root, split, "adjacent", stem)
    prompt = _choose_prompt(stem, span_type="adjacent")
    metadata = {
        "dataset": "finevideo",
        "span_type": "adjacent",
        "source_video_path": relativize_video_path(str(source_video)),
        "scene_index_start": start_scene_index,
        "scene_index_end": start_scene_index + window_size - 1,
        "scene_window_size": window_size,
        "scene_ids": [scene.get("sceneId") for scene in window],
    }
    return SpanCandidate(
        sample_id=stem,
        split=split,
        dataset="finevideo",
        span_type="adjacent",
        source_video_abs=str(source_video),
        output_video_abs=str(output_path),
        clip_start_sec=first_start,
        clip_end_sec=last_end,
        prompt=prompt,
        target_text=target,
        metadata=metadata,
    )


def _build_full_candidate(
    *,
    split: str,
    source_video: Path,
    source_stem: str,
    content_metadata: dict[str, Any],
    youtube_title: str,
) -> SpanCandidate | None:
    target = _full_target_from_description(content_metadata)
    if not target:
        return None
    stem = _span_stem(split, source_stem, "full", "full")
    prompt = _choose_prompt(stem, span_type="full")
    metadata = {
        "dataset": "finevideo",
        "span_type": "full",
        "source_video_path": relativize_video_path(str(source_video)),
        "youtube_title": youtube_title,
    }
    return SpanCandidate(
        sample_id=stem,
        split=split,
        dataset="finevideo",
        span_type="full",
        source_video_abs=str(source_video),
        output_video_abs=str(source_video),
        clip_start_sec=None,
        clip_end_sec=None,
        prompt=prompt,
        target_text=target,
        metadata=metadata,
    )


def collect_finevideo_span_pools(
    split_root: str | Path,
    *,
    split: str,
    clips_root: str | Path,
) -> dict[str, list[SpanCandidate]]:
    split_root = Path(split_root)
    clips_root = Path(clips_root)
    scene_pool: list[SpanCandidate] = []
    adjacent_pool: list[SpanCandidate] = []
    full_pool: list[SpanCandidate] = []

    json_paths = sorted(split_root.rglob("*.json"))
    for metadata_path in json_paths:
        video_path = metadata_path.with_suffix(".mp4")
        if not video_path.exists():
            continue
        metadata = _read_json(metadata_path)
        content_metadata = metadata.get("content_metadata") or {}
        scenes = content_metadata.get("scenes") or []
        if not isinstance(scenes, list) or not scenes:
            continue

        source_stem = metadata_path.stem
        youtube_title = str(metadata.get("youtube_title") or "")

        for scene_index, scene in enumerate(scenes):
            if not isinstance(scene, dict):
                continue
            candidate = _build_scene_candidate(
                split=split,
                source_video=video_path,
                source_stem=source_stem,
                scene=scene,
                scene_index=scene_index,
                clips_root=clips_root,
            )
            if candidate is not None:
                scene_pool.append(candidate)

        for window_size in (2, 3):
            for start_scene_index in range(0, len(scenes) - window_size + 1):
                candidate = _build_adjacent_candidate(
                    split=split,
                    source_video=video_path,
                    source_stem=source_stem,
                    scenes=scenes,
                    start_scene_index=start_scene_index,
                    window_size=window_size,
                    clips_root=clips_root,
                )
                if candidate is not None:
                    adjacent_pool.append(candidate)

        full_candidate = _build_full_candidate(
            split=split,
            source_video=video_path,
            source_stem=source_stem,
            content_metadata=content_metadata,
            youtube_title=youtube_title,
        )
        if full_candidate is not None:
            full_pool.append(full_candidate)

    return {
        "scene": scene_pool,
        "adjacent": adjacent_pool,
        "full": full_pool,
    }


def _counts_from_target(
    target_total: int,
    *,
    scene_ratio: float,
    adjacent_ratio: float,
    full_ratio: float,
) -> tuple[int, int, int]:
    scene_count = int(round(target_total * scene_ratio))
    adjacent_count = int(round(target_total * adjacent_ratio))
    full_count = target_total - scene_count - adjacent_count
    return scene_count, adjacent_count, full_count


def max_unique_mixture_total(
    *,
    scene_pool_size: int,
    adjacent_pool_size: int,
    full_pool_size: int,
    scene_ratio: float,
    adjacent_ratio: float,
    full_ratio: float,
) -> int:
    by_scene = math.floor(scene_pool_size / scene_ratio) if scene_ratio > 0 else math.inf
    by_adjacent = (
        math.floor(adjacent_pool_size / adjacent_ratio) if adjacent_ratio > 0 else math.inf
    )
    by_full = math.floor(full_pool_size / full_ratio) if full_ratio > 0 else math.inf
    return int(min(by_scene, by_adjacent, by_full))


def sample_mixture(
    pools: dict[str, list[SpanCandidate]],
    *,
    target_total: int,
    scene_ratio: float = 0.60,
    adjacent_ratio: float = 0.30,
    full_ratio: float = 0.10,
    seed: int = 42,
) -> list[SpanCandidate]:
    if target_total <= 0:
        target_total = max_unique_mixture_total(
            scene_pool_size=len(pools["scene"]),
            adjacent_pool_size=len(pools["adjacent"]),
            full_pool_size=len(pools["full"]),
            scene_ratio=scene_ratio,
            adjacent_ratio=adjacent_ratio,
            full_ratio=full_ratio,
        )
    scene_count, adjacent_count, full_count = _counts_from_target(
        target_total,
        scene_ratio=scene_ratio,
        adjacent_ratio=adjacent_ratio,
        full_ratio=full_ratio,
    )
    if scene_count > len(pools["scene"]):
        raise ValueError(
            f"scene pool too small for target: need {scene_count}, have {len(pools['scene'])}"
        )
    if adjacent_count > len(pools["adjacent"]):
        raise ValueError(
            f"adjacent pool too small for target: need {adjacent_count}, have {len(pools['adjacent'])}"
        )
    if full_count > len(pools["full"]):
        raise ValueError(
            f"full pool too small for target: need {full_count}, have {len(pools['full'])}"
        )

    rng = random.Random(seed)
    out: list[SpanCandidate] = []
    out.extend(rng.sample(pools["scene"], scene_count))
    out.extend(rng.sample(pools["adjacent"], adjacent_count))
    out.extend(rng.sample(pools["full"], full_count))
    rng.shuffle(out)
    return out


def sample_fixed_count(pool: list[SpanCandidate], *, count: int, seed: int) -> list[SpanCandidate]:
    if count <= 0:
        return []
    if count >= len(pool):
        return list(pool)
    rng = random.Random(seed)
    return rng.sample(pool, count)


def _extract_one(
    candidate: SpanCandidate,
    overwrite: bool,
    ffmpeg_threads: int,
    ffmpeg_preset: str,
    ffmpeg_crf: int,
    include_audio: bool,
) -> tuple[str, str | None]:
    if candidate.span_type == "full":
        return candidate.sample_id, None
    if candidate.clip_start_sec is None or candidate.clip_end_sec is None:
        return candidate.sample_id, "missing_clip_bounds"
    try:
        _ffmpeg_extract_clip(
            Path(candidate.source_video_abs),
            Path(candidate.output_video_abs),
            candidate.clip_start_sec,
            candidate.clip_end_sec,
            overwrite=overwrite,
            ffmpeg_threads=ffmpeg_threads,
            ffmpeg_preset=ffmpeg_preset,
            ffmpeg_crf=ffmpeg_crf,
            include_audio=include_audio,
        )
        return candidate.sample_id, None
    except Exception as e:  # pylint: disable=broad-except
        return candidate.sample_id, str(e)


def materialize_span_clips(
    candidates: list[SpanCandidate],
    *,
    num_workers: int = 16,
    overwrite: bool = False,
    ffmpeg_threads: int = 1,
    ffmpeg_preset: str = "ultrafast",
    ffmpeg_crf: int = 30,
    include_audio: bool = False,
) -> list[tuple[str, str]]:
    clip_candidates = [candidate for candidate in candidates if candidate.span_type != "full"]
    failures: list[tuple[str, str]] = []
    if not clip_candidates:
        return failures
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _extract_one,
                candidate,
                overwrite,
                ffmpeg_threads,
                ffmpeg_preset,
                ffmpeg_crf,
                include_audio,
            )
            for candidate in clip_candidates
        ]
        done = 0
        for future in as_completed(futures):
            sample_id, error = future.result()
            done += 1
            if error is not None:
                failures.append((sample_id, error))
            if done % 500 == 0:
                print(
                    f"[clip] done={done}/{len(clip_candidates)} failures={len(failures)}",
                    flush=True,
                )
    return failures


def candidates_to_rows(candidates: list[SpanCandidate]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        video_rel = relativize_video_path(candidate.output_video_abs)
        row = {
            "id": candidate.sample_id,
            "video_path": video_rel,
            "task_type": "caption",
            "prompt": candidate.prompt,
            "target_text": candidate.target_text,
            "dataset": candidate.dataset,
            "split": candidate.split,
            "metadata": dict(candidate.metadata),
        }
        if candidate.clip_start_sec is not None:
            row["clip_start_sec"] = candidate.clip_start_sec
        if candidate.clip_end_sec is not None:
            row["clip_end_sec"] = candidate.clip_end_sec
        rows.append(normalize_manifest_row(row, default_split=candidate.split))
    return rows


def write_finevideo_manifests(
    *,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    val_scene_rows: list[dict[str, Any]],
    val_adjacent_rows: list[dict[str, Any]],
    val_full_rows: list[dict[str, Any]],
    val_gen_rows: list[dict[str, Any]],
    train_out: str | Path,
    val_out: str | Path,
    val_scene_out: str | Path,
    val_adjacent_out: str | Path,
    val_full_out: str | Path,
    val_core_out: str | Path,
    val_gen_out: str | Path,
) -> None:
    dump_jsonl(train_out, train_rows)
    dump_jsonl(val_out, val_rows)
    dump_jsonl(val_scene_out, val_scene_rows)
    dump_jsonl(val_adjacent_out, val_adjacent_rows)
    dump_jsonl(val_full_out, val_full_rows)
    dump_jsonl(val_core_out, val_rows[:1024])
    dump_jsonl(val_gen_out, val_gen_rows)


def ensure_relative_to_data_root(path: str | Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        return str(path)
    return str(path.relative_to(DATA_ROOT))
