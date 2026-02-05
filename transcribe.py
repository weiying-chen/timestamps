#!/usr/bin/env python3
import argparse
import os
import time
import subprocess
from faster_whisper import WhisperModel

FPS = 30


def probe_duration_seconds(path: str) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def format_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def normalize_segments(segments):
    for seg in segments:
        words = getattr(seg, "words", None)
        if words:
            start = next((w.start for w in words if w.start is not None), seg.start)
            end = next((w.end for w in reversed(words) if w.end is not None), seg.end)
            if start is not None and end is not None and end >= start:
                seg.start = start
                seg.end = end
        yield seg


def write_srt(segments, output_path: str, show_progress: bool, total_seconds: float | None, progress_interval: float) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        last_print = time.monotonic()
        count = 0
        for seg in segments:
            count += 1
            f.write(
                f"{count}\n{format_ts(seg.start)} --> {format_ts(seg.end)}\n{seg.text.strip()}\n\n"
            )
            if show_progress:
                now = time.monotonic()
                if now - last_print >= progress_interval:
                    if total_seconds:
                        pct = min(100.0, (seg.end / total_seconds) * 100.0)
                        print(f"[progress] {pct:5.1f}%  {seg.end:.1f}s / {total_seconds:.1f}s")
                    else:
                        print(f"[progress] segments={count} last_end={seg.end:.1f}s")
                    last_print = now


def format_tc(seconds: float) -> str:
    total_frames = int(round(seconds * FPS))
    h = total_frames // (FPS * 3600)
    total_frames %= FPS * 3600
    m = total_frames // (FPS * 60)
    total_frames %= FPS * 60
    s = total_frames // FPS
    f = total_frames % FPS
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def write_tsv(segments, output_path: str, show_progress: bool, total_seconds: float | None, progress_interval: float) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        last_print = time.monotonic()
        count = 0
        for seg in segments:
            count += 1
            start = format_tc(seg.start)
            end = format_tc(seg.end)
            text = seg.text.strip()
            f.write(f"{start}\t{end}\n")
            f.write(f"{text}\n\n")
            if show_progress:
                now = time.monotonic()
                if now - last_print >= progress_interval:
                    if total_seconds:
                        pct = min(100.0, (seg.end / total_seconds) * 100.0)
                        print(f"[progress] {pct:5.1f}%  {seg.end:.1f}s / {total_seconds:.1f}s")
                    else:
                        print(f"[progress] segments={count} last_end={seg.end:.1f}s")
                    last_print = now


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio with faster-whisper.")
    parser.add_argument("input", help="Path to audio file (e.g., input.mp3)")
    parser.add_argument("--model", default="medium", help="Model size (tiny/base/small/medium/large-v3)")
    parser.add_argument("--compute-type", default="int8", help="Compute type (int8/float32/float16)")
    parser.add_argument("--language", default="zh", help="Language code (e.g., zh, zh-tw)")
    parser.add_argument("--output", default=None, help="Output SRT path")
    parser.add_argument("--format", default="srt", choices=["srt", "tsv"], help="Output format")
    parser.add_argument("--progress", action="store_true", help="Print progress while transcribing")
    parser.add_argument("--progress-interval", type=float, default=5.0, help="Seconds between progress updates")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding")
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Use word-level timestamps to set segment boundaries (output format unchanged)",
    )

    args = parser.parse_args()

    output_path = args.output
    if not output_path:
        base = os.path.splitext(os.path.basename(args.input))[0]
        ext = "srt" if args.format == "srt" else "tsv"
        output_path = f"{base}.{ext}"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    model = WhisperModel(args.model, device="cpu", compute_type=args.compute_type)
    segments, _info = model.transcribe(
        args.input,
        language=args.language,
        beam_size=args.beam_size,
        word_timestamps=args.word_timestamps,
    )

    total_seconds = probe_duration_seconds(args.input) if args.progress else None
    if args.word_timestamps:
        segments = normalize_segments(segments)
    if args.format == "srt":
        write_srt(segments, output_path, show_progress=args.progress, total_seconds=total_seconds, progress_interval=args.progress_interval)
    else:
        write_tsv(segments, output_path, show_progress=args.progress, total_seconds=total_seconds, progress_interval=args.progress_interval)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
