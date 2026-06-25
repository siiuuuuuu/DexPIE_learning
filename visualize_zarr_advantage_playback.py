import argparse
import json
import os
import time

import cv2
import numpy as np
import zarr


WINDOW_NAME = "Zarr Advantage Viewer"
WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 920
HEADER_HEIGHT = 92
GRAPH_HEIGHT = 245
FOOTER_HEIGHT = 74
MARGIN = 20
PANEL_GAP = 16
DEFAULT_FPS = 25.0

COLOR_BACKGROUND = (18, 20, 25)
COLOR_HEADER = (25, 28, 35)
COLOR_PANEL = (30, 34, 42)
COLOR_PANEL_BORDER = (58, 64, 76)
COLOR_GRID = (48, 53, 63)
COLOR_TEXT = (238, 241, 246)
COLOR_MUTED = (157, 164, 178)
COLOR_ACCENT = (235, 164, 52)
COLOR_SUCCESS = (91, 201, 125)
COLOR_WARNING = (62, 190, 245)
COLOR_AUTONOMOUS = (82, 139, 91)
COLOR_INTERVENTION = (60, 74, 211)
COLOR_ZERO = (105, 115, 135)
COLOR_THRESHOLD = (80, 170, 230)
COLOR_PLAYHEAD = (250, 250, 250)


def _draw_text(image, text, origin, scale, color, thickness=1):
    x, y = origin
    cv2.putText(
        image,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _fit_text(text, max_width, scale, thickness=1):
    if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] <= max_width:
        return text

    suffix = "..."
    text = str(text)
    while text:
        candidate = text + suffix
        width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if width <= max_width:
            return candidate
        text = text[:-1]
    return suffix


def _to_bgr(frame):
    image = np.asarray(frame)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))

    if image.dtype != np.uint8:
        image = np.nan_to_num(image)
        if np.issubdtype(image.dtype, np.floating) and image.size and image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError(f"unsupported frame shape: {image.shape}")
    if image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    raise ValueError(f"unsupported channel count: {image.shape[2]}")


def _place_fitted_image(canvas, image, x, y, width, height):
    image_height, image_width = image.shape[:2]
    scale = min(width / image_width, height / image_height)
    target_width = max(1, int(round(image_width * scale)))
    target_height = max(1, int(round(image_height * scale)))
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    image_x = x + (width - target_width) // 2
    image_y = y + (height - target_height) // 2
    canvas[image_y : image_y + target_height, image_x : image_x + target_width] = resized


def _draw_camera_panel(canvas, frame, rect, title, dataset_name):
    x, y, width, height = rect
    cv2.rectangle(canvas, (x, y), (x + width, y + height), COLOR_PANEL, -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), COLOR_PANEL_BORDER, 1)

    label_height = 44
    cv2.line(canvas, (x, y + label_height), (x + width, y + label_height), COLOR_PANEL_BORDER, 1)
    _draw_text(canvas, title, (x + 16, y + 29), 0.61, COLOR_TEXT, 2)
    _draw_text(canvas, dataset_name, (x + 16, y + height - 12), 0.46, COLOR_MUTED, 1)

    image_x = x + 12
    image_y = y + label_height + 12
    image_width = width - 24
    image_height = height - label_height - 40

    if frame is None:
        message = "CAMERA STREAM NOT AVAILABLE"
        message_width = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0][0]
        _draw_text(
            canvas,
            message,
            (x + (width - message_width) // 2, image_y + image_height // 2),
            0.62,
            COLOR_WARNING,
            2,
        )
        return

    try:
        image = _to_bgr(frame)
        resolution = f"{image.shape[1]} x {image.shape[0]}"
        resolution_width = cv2.getTextSize(resolution, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0][0]
        _draw_text(canvas, resolution, (x + width - resolution_width - 16, y + 28), 0.46, COLOR_MUTED, 1)
        _place_fitted_image(canvas, image, image_x, image_y, image_width, image_height)
    except ValueError as exc:
        message = _fit_text(str(exc).upper(), image_width - 20, 0.52, 1)
        _draw_text(canvas, message, (image_x + 10, image_y + image_height // 2), 0.52, COLOR_WARNING, 1)


def _format_time(seconds):
    total_seconds = max(0, int(seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _as_bool_array(array, length):
    if array is None:
        return np.zeros(length, dtype=bool)
    result = np.asarray(array, dtype=bool).reshape(-1)
    if len(result) < length:
        result = np.pad(result, (0, length - len(result)), constant_values=False)
    elif len(result) > length:
        result = result[:length]
    return result


def _safe_get(root, group_name, key):
    if group_name in root and key in root[group_name]:
        return root[group_name][key]
    path = f"{group_name}/{key}"
    try:
        return root[path]
    except Exception:
        return None


def _get_episode_bounds(episode_ends, episode_idx):
    if episode_idx < 0 or episode_idx >= len(episode_ends):
        raise IndexError(f"episode index {episode_idx} out of range [0, {len(episode_ends) - 1}]")
    start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end = int(episode_ends[episode_idx])
    return start, end


def _load_thresholds(path):
    if path is None:
        return []
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        payload = json.load(f)

    thresholds = []
    for item in payload.get("advantage_quantiles", []):
        if not isinstance(item, dict):
            continue
        if "quantile" not in item or "advantage_threshold" not in item:
            continue
        thresholds.append((float(item["quantile"]), float(item["advantage_threshold"])))
    return thresholds


def _value_to_y(value, value_min, value_max, y, height):
    ratio = (float(value) - value_min) / (value_max - value_min)
    ratio = min(1.0, max(0.0, ratio))
    return int(round(y + height - 1 - ratio * (height - 1)))


def _draw_intervention_strip(canvas, rect, intervention):
    x, y, width, height = rect
    if intervention is None or len(intervention) == 0:
        return
    strip = np.empty((1, len(intervention), 3), dtype=np.uint8)
    strip[0, ~intervention] = COLOR_AUTONOMOUS
    strip[0, intervention] = COLOR_INTERVENTION
    strip = cv2.resize(strip, (width, height), interpolation=cv2.INTER_NEAREST)
    canvas[y : y + height, x : x + width] = strip
    cv2.rectangle(canvas, (x, y), (x + width, y + height), COLOR_PANEL_BORDER, 1)


def _draw_advantage_graph(canvas, rect, advantages, frame_idx, thresholds, intervention):
    x, y, width, height = rect
    cv2.rectangle(canvas, (x, y), (x + width, y + height), COLOR_PANEL, -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), COLOR_PANEL_BORDER, 1)
    _draw_text(canvas, "ADVANTAGE", (x + 16, y + 30), 0.62, COLOR_TEXT, 2)

    plot_x = x + 56
    plot_y = y + 48
    plot_w = width - 82
    plot_h = height - 86
    strip_y = y + height - 24
    strip_h = 10

    values = np.asarray(advantages, dtype=np.float32).reshape(-1)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        finite = np.asarray([0.0], dtype=np.float32)

    extra_values = [thr for _, thr in thresholds]
    value_min = float(min(np.min(finite), *(extra_values or [np.min(finite)]), 0.0))
    value_max = float(max(np.max(finite), *(extra_values or [np.max(finite)]), 0.0))
    if value_max <= value_min:
        value_min -= 1.0
        value_max += 1.0
    padding = max(1e-6, (value_max - value_min) * 0.08)
    value_min -= padding
    value_max += padding

    for i in range(5):
        gy = plot_y + int(round(i * plot_h / 4.0))
        cv2.line(canvas, (plot_x, gy), (plot_x + plot_w, gy), COLOR_GRID, 1)
        label_value = value_max - (value_max - value_min) * (i / 4.0)
        _draw_text(canvas, f"{label_value:.3f}", (x + 10, gy + 5), 0.36, COLOR_MUTED, 1)

    total = len(values)
    if total <= 1:
        total = 2

    def frame_to_x(index):
        return int(round(plot_x + (index / (total - 1)) * plot_w))

    if value_min <= 0.0 <= value_max:
        zero_y = _value_to_y(0.0, value_min, value_max, plot_y, plot_h)
        cv2.line(canvas, (plot_x, zero_y), (plot_x + plot_w, zero_y), COLOR_ZERO, 1, cv2.LINE_AA)
        _draw_text(canvas, "0", (plot_x + plot_w + 6, zero_y + 4), 0.36, COLOR_MUTED, 1)

    for quantile, threshold in thresholds:
        if value_min <= threshold <= value_max:
            ty = _value_to_y(threshold, value_min, value_max, plot_y, plot_h)
            cv2.line(canvas, (plot_x, ty), (plot_x + plot_w, ty), COLOR_THRESHOLD, 1, cv2.LINE_AA)
            _draw_text(canvas, f"q{quantile:.1f}", (plot_x + plot_w + 6, ty + 4), 0.36, COLOR_THRESHOLD, 1)

    max_idx = min(frame_idx, len(values) - 1)
    if max_idx >= 0:
        pts = []
        for idx in range(max_idx + 1):
            value = float(values[idx])
            if not np.isfinite(value):
                continue
            pts.append((frame_to_x(idx), _value_to_y(value, value_min, value_max, plot_y, plot_h)))
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False, COLOR_ACCENT, 2, cv2.LINE_AA)
        elif len(pts) == 1:
            cv2.circle(canvas, pts[0], 3, COLOR_ACCENT, -1, cv2.LINE_AA)

    playhead_x = frame_to_x(max_idx)
    cv2.line(canvas, (playhead_x, plot_y), (playhead_x, plot_y + plot_h), COLOR_PLAYHEAD, 1, cv2.LINE_AA)
    if 0 <= max_idx < len(values) and np.isfinite(values[max_idx]):
        point_y = _value_to_y(float(values[max_idx]), value_min, value_max, plot_y, plot_h)
        cv2.circle(canvas, (playhead_x, point_y), 5, COLOR_ACCENT, -1, cv2.LINE_AA)
        cv2.circle(canvas, (playhead_x, point_y), 8, COLOR_PLAYHEAD, 1, cv2.LINE_AA)
        current_text = f"frame {max_idx + 1}/{len(values)}  advantage {float(values[max_idx]):.6f}"
    else:
        current_text = f"frame {max_idx + 1}/{len(values)}  advantage nan"
    _draw_text(canvas, current_text, (plot_x, y + height - 42), 0.50, COLOR_TEXT, 1)

    _draw_intervention_strip(canvas, (plot_x, strip_y, plot_w, strip_h), intervention)


def compose_dashboard(
    front_frame,
    wrist_frame,
    zarr_name,
    episode_idx,
    episode_count,
    frame_idx,
    total_frames,
    fps,
    advantages,
    thresholds,
    intervention=None,
    paused=False,
    warning=None,
):
    canvas = np.full((WINDOW_HEIGHT, WINDOW_WIDTH, 3), COLOR_BACKGROUND, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (WINDOW_WIDTH, HEADER_HEIGHT), COLOR_HEADER, -1)

    _draw_text(canvas, "ZARR ADVANTAGE PLAYBACK", (MARGIN, 37), 0.84, COLOR_TEXT, 2)
    file_text = _fit_text(zarr_name, WINDOW_WIDTH - 420, 0.50, 1)
    _draw_text(canvas, file_text, (MARGIN, 67), 0.50, COLOR_MUTED, 1)
    if warning:
        warning_text = _fit_text(warning, WINDOW_WIDTH - 420, 0.42, 1)
        _draw_text(canvas, warning_text, (MARGIN, 88), 0.42, COLOR_WARNING, 1)

    status_text = "PAUSED" if paused else "PLAYING"
    status_color = COLOR_WARNING if paused else COLOR_SUCCESS
    badge_width = 130
    badge_height = 38
    badge_x = WINDOW_WIDTH - MARGIN - badge_width
    badge_y = 27
    cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_width, badge_y + badge_height), status_color, -1)
    status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
    _draw_text(
        canvas,
        status_text,
        (badge_x + (badge_width - status_size[0]) // 2, badge_y + (badge_height + status_size[1]) // 2),
        0.55,
        COLOR_BACKGROUND,
        2,
    )

    graph_y = WINDOW_HEIGHT - FOOTER_HEIGHT - GRAPH_HEIGHT
    panel_y = HEADER_HEIGHT + MARGIN
    panel_height = graph_y - panel_y - MARGIN
    panel_width = (WINDOW_WIDTH - 2 * MARGIN - PANEL_GAP) // 2
    left_rect = (MARGIN, panel_y, panel_width, panel_height)
    right_rect = (MARGIN + panel_width + PANEL_GAP, panel_y, panel_width, panel_height)
    _draw_camera_panel(canvas, front_frame, left_rect, "FRONT CAMERA", "data/img")
    _draw_camera_panel(canvas, wrist_frame, right_rect, "WRIST CAMERA", "data/wrist_img")

    graph_rect = (MARGIN, graph_y, WINDOW_WIDTH - 2 * MARGIN, GRAPH_HEIGHT - MARGIN)
    _draw_advantage_graph(canvas, graph_rect, advantages, frame_idx, thresholds, intervention)

    footer_y = WINDOW_HEIGHT - FOOTER_HEIGHT
    cv2.rectangle(canvas, (0, footer_y), (WINDOW_WIDTH, WINDOW_HEIGHT), COLOR_HEADER, -1)
    current_time = _format_time(frame_idx / fps)
    total_time = _format_time(total_frames / fps)
    intervention_steps = int(np.count_nonzero(intervention)) if intervention is not None else 0
    playback_info = (
        f"Episode {episode_idx + 1}/{episode_count}    "
        f"Frame {frame_idx + 1:,}/{total_frames:,}    "
        f"{current_time}/{total_time}    {fps:g} FPS    "
        f"Intervention {intervention_steps:,}"
    )
    _draw_text(canvas, playback_info, (MARGIN, footer_y + 29), 0.52, COLOR_TEXT, 1)
    controls = "SPACE Pause/Resume     F/B +/-10 frames     R Restart     N/P Episode     Q Quit"
    _draw_text(canvas, controls, (MARGIN, footer_y + 58), 0.43, COLOR_MUTED, 1)
    return canvas


def play_episode(root, args, episode_idx, thresholds):
    image_data = _safe_get(root, "data", args.image_key)
    wrist_data = _safe_get(root, "data", args.wrist_key)
    advantage_data = _safe_get(root, "data", args.advantage_key)
    intervention_data = _safe_get(root, "data", args.intervention_key)
    episode_ends = _safe_get(root, "meta", "episode_ends")

    if image_data is None:
        raise KeyError(f"missing data/{args.image_key}")
    if advantage_data is None:
        raise KeyError(f"missing data/{args.advantage_key}")
    if episode_ends is None:
        raise KeyError("missing meta/episode_ends")

    episode_ends = np.asarray(episode_ends[:], dtype=np.int64).reshape(-1)
    start, end = _get_episode_bounds(episode_ends, episode_idx)
    episode_count = len(episode_ends)
    episode_length = end - start
    if episode_length <= 0:
        raise ValueError(f"episode {episode_idx} is empty: start={start}, end={end}")

    total_frames = min(episode_length, len(image_data) - start, len(advantage_data) - start)
    warnings = []
    if wrist_data is None:
        warnings.append(f"data/{args.wrist_key} unavailable")
    else:
        total_frames = min(total_frames, len(wrist_data) - start)
    if total_frames <= 0:
        raise ValueError(
            f"episode {episode_idx} has no synchronized frames after alignment: "
            f"start={start}, end={end}"
        )
    if total_frames < episode_length:
        warnings.append(f"trimmed episode from {episode_length} to {total_frames} synchronized frames")

    advantages = np.asarray(advantage_data[start : start + total_frames], dtype=np.float32).reshape(-1)
    intervention = None
    if intervention_data is not None:
        intervention = _as_bool_array(intervention_data[start : start + total_frames], total_frames)

    zarr_name = os.path.basename(os.path.abspath(os.path.expanduser(args.zarr_path)))
    warning = "; ".join(warnings) if warnings else None
    print(f"\nPlaying episode {episode_idx + 1}/{episode_count}: frames={total_frames}")
    print(f"  zarr: {os.path.abspath(os.path.expanduser(args.zarr_path))}")
    print(f"  image: data/{args.image_key} {image_data.shape}")
    print(f"  wrist: {None if wrist_data is None else wrist_data.shape}")
    print(f"  advantage: data/{args.advantage_key} {advantage_data.shape}")
    if intervention is not None:
        print(f"  intervention steps: {int(np.count_nonzero(intervention))}/{len(intervention)}")
    print("  Controls: SPACE pause/resume, F/B seek, R restart, N/P episode, Q quit")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    paused = False
    frame_idx = int(np.clip(args.start_frame, 0, total_frames - 1))
    frame_delay_ms = max(1, int(round(1000.0 / args.fps)))

    while frame_idx < total_frames:
        loop_start = time.monotonic()
        global_idx = start + frame_idx
        front_frame = image_data[global_idx]
        wrist_frame = None if wrist_data is None else wrist_data[global_idx]
        dashboard = compose_dashboard(
            front_frame=front_frame,
            wrist_frame=wrist_frame,
            zarr_name=zarr_name,
            episode_idx=episode_idx,
            episode_count=episode_count,
            frame_idx=frame_idx,
            total_frames=total_frames,
            fps=args.fps,
            advantages=advantages,
            thresholds=thresholds,
            intervention=intervention,
            paused=paused,
            warning=warning,
        )
        cv2.imshow(WINDOW_NAME, dashboard)

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return "quit", episode_idx

        render_ms = int((time.monotonic() - loop_start) * 1000)
        wait_ms = 30 if paused else max(1, frame_delay_ms - render_ms)
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord("q"):
            return "quit", episode_idx
        if key == ord("n"):
            return "episode", min(episode_idx + 1, episode_count - 1)
        if key == ord("p"):
            return "episode", max(episode_idx - 1, 0)
        if key == ord(" "):
            paused = not paused
            continue
        if key == ord("r"):
            frame_idx = 0
            paused = False
            continue
        if key == ord("f"):
            frame_idx = min(frame_idx + 10, total_frames - 1)
            continue
        if key == ord("b"):
            frame_idx = max(frame_idx - 10, 0)
            continue
        if not paused:
            frame_idx += 1

    return "episode", min(episode_idx + 1, episode_count - 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play a zarr episode while drawing the advantage curve up to the current frame."
    )
    parser.add_argument("zarr_path", help="Path to the zarr dataset.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to start from, 0-based.")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame index to start from inside the episode.")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Playback frame rate.")
    parser.add_argument("--image-key", default="img", help="Image key under data/.")
    parser.add_argument("--wrist-key", default="wrist_img", help="Wrist image key under data/.")
    parser.add_argument("--advantage-key", default="advantage", help="Advantage key under data/.")
    parser.add_argument("--intervention-key", default="intervention", help="Intervention key under data/.")
    parser.add_argument(
        "--advantage-quantiles-json",
        default=None,
        help="Optional advantage_quantiles.json; draws its quantile thresholds.",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be greater than zero")
    return args


def main():
    args = parse_args()
    args.zarr_path = os.path.abspath(os.path.expanduser(args.zarr_path))
    root = zarr.open(args.zarr_path, mode="r")
    episode_ends = _safe_get(root, "meta", "episode_ends")
    if episode_ends is None:
        raise KeyError("missing meta/episode_ends")

    episode_count = len(episode_ends)
    if episode_count == 0:
        raise ValueError("dataset has no episodes")
    episode_idx = int(np.clip(args.episode, 0, episode_count - 1))
    thresholds = _load_thresholds(args.advantage_quantiles_json)
    episode_ends_np = np.asarray(episode_ends[:], dtype=np.int64).reshape(-1)

    print(f"Found {episode_count} episode(s) in: {args.zarr_path}")
    for idx in range(episode_count):
        start, end = _get_episode_bounds(episode_ends_np, idx)
        marker = "*" if idx == episode_idx else " "
        print(f"{marker} episode {idx}: frames {end - start} [{start}, {end})")

    try:
        while True:
            result, next_episode_idx = play_episode(root, args, episode_idx, thresholds)
            if result == "quit":
                break
            if next_episode_idx == episode_idx:
                break
            episode_idx = next_episode_idx
            args.start_frame = 0
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
