import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg


def _normalize_frame_float(frame_raw):
    frame_float = frame_raw.astype(np.float32)
    if frame_float.max() > 1.0:
        frame_float /= 255.0
    return np.clip(frame_float, 0.0, 1.0)


def create_combined_visualization(high_res_frame, attribution_overlay, attribution_graph,
                                   prediction_text, ground_truth_text, frame_number):
    """Create a combined visualization with high-res video, attribution overlay, graph, and labels."""
    hr_h, hr_w = high_res_frame.shape[:2]
    attribution_overlay_resized = cv2.resize(attribution_overlay, (448, 448))

    canvas = FigureCanvasAgg(attribution_graph)
    canvas.draw()
    graph_image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    graph_image = graph_image.reshape(canvas.get_width_height()[::-1] + (4,))
    graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGBA2RGB)

    graph_h, graph_w = graph_image.shape[:2]
    new_graph_width = 448
    new_graph_height = int(graph_h * new_graph_width / graph_w)
    graph_resized = cv2.resize(graph_image, (new_graph_width, new_graph_height))

    right_column = np.vstack([attribution_overlay_resized, graph_resized])
    right_h = right_column.shape[0]
    if right_h < hr_h:
        padding = np.ones((hr_h - right_h, right_column.shape[1], 3), dtype=np.uint8) * 255
        right_column = np.vstack([right_column, padding])
    elif right_h > hr_h:
        right_column = right_column[:hr_h, :, :]

    combined_main = np.hstack([high_res_frame, right_column])
    return combined_main


def _render_text_free_graph(attribution_scores, current_idx, panel_width, panel_height=180):
    fig, ax = plt.subplots(figsize=(panel_width / 100, panel_height / 100), dpi=100)
    frame_numbers = np.arange(len(attribution_scores))
    ax.plot(frame_numbers, attribution_scores, linewidth=2, color="darkorange")
    ax.scatter([current_idx], [attribution_scores[current_idx]], s=60, color="white", zorder=5)
    ax.set_xlim(0, max(len(attribution_scores) - 1, 1))

    score_min = float(np.min(attribution_scores))
    score_max = float(np.max(attribution_scores))
    if score_max > score_min:
        pad = (score_max - score_min) * 0.1
        ax.set_ylim(score_min - pad, score_max + pad)

    ax.grid(True, alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.2)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    graph_img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    graph_img = graph_img.reshape(canvas.get_width_height()[::-1] + (4,))
    graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    return graph_img


def save_combined_video(video_directory, video_name, high_res_video_file,
                        attribution_scores, prediction_text, ground_truth_text,
                        keywords, visualizations):
    """Create and save a combined visualization video."""
    cap = cv2.VideoCapture(high_res_video_file)
    if not cap.isOpened():
        print(f"Error: Could not open high-res video {high_res_video_file}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_per_attribution = int(fps)

    output_path = os.path.join(video_directory, f"{video_name}_combined_visualization.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None

    attribution_graphs = []
    for idx in range(30):
        plt.figure(figsize=(6, 3))
        frame_numbers = np.arange(len(attribution_scores))
        plt.plot(frame_numbers, attribution_scores, marker="o", linewidth=2, markersize=4, color="darkorange")
        plt.plot(idx, attribution_scores[idx], marker="o", markersize=10, color="white", zorder=5)
        plt.xlabel("Frame Number")
        plt.ylabel("Attribution Score")
        plt.title(f"Attribution Scores - {ground_truth_text}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()
        graph_image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        graph_image = graph_image.reshape(canvas.get_width_height()[::-1] + (4,))
        graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGBA2RGB)
        attribution_graphs.append(graph_image)
        plt.close()

    frame_count = 0
    while True:
        ret, high_res_frame = cap.read()
        if not ret:
            break

        high_res_frame = cv2.cvtColor(high_res_frame, cv2.COLOR_BGR2RGB)
        current_attribution_idx = min(frame_count // frames_per_attribution, 29)
        if current_attribution_idx >= 30:
            break

        attribution_overlay = visualizations[current_attribution_idx]
        attribution_graph_image = attribution_graphs[current_attribution_idx]

        attribution_overlay_resized = cv2.resize(attribution_overlay, (448, 448))

        graph_h, graph_w = attribution_graph_image.shape[:2]
        new_graph_width = 448
        new_graph_height = int(graph_h * new_graph_width / graph_w)
        graph_resized = cv2.resize(attribution_graph_image, (new_graph_width, new_graph_height))

        hr_h, hr_w = high_res_frame.shape[:2]
        right_column = np.vstack([attribution_overlay_resized, graph_resized])

        right_h = right_column.shape[0]
        if right_h < hr_h:
            padding = np.ones((hr_h - right_h, right_column.shape[1], 3), dtype=np.uint8) * 255
            right_column = np.vstack([right_column, padding])
        elif right_h > hr_h:
            right_column = right_column[:hr_h, :, :]

        combined_main = np.hstack([high_res_frame, right_column])
        combined_frame = combined_main

        if out is None:
            height, width = combined_frame.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        combined_frame_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
        out.write(combined_frame_bgr)

        frame_count += 1

    cap.release()
    if out is not None:
        out.release()
        print(f"Combined visualization saved to: {output_path}")
    else:
        print("Error: Could not create output video")


def save_lowres_visualization_video(video_directory, video_name, lowres_frames,
                                    attribution_maps=None, attribution_scores=None,
                                    prediction_text='', ground_truth_text='',
                                    attribution_evaluator=None, matched_video_id=None,
                                    cam_threshold=0.5, attribution_method_name='Attribution',
                                    attribution_renderer=None,
                                    draw_overlay_bboxes=False,
                                    render_text=False,
                                    **legacy_kwargs):
    """Create and save a visualization video using low-quality dataset frames."""
    if attribution_maps is None and "gradcam_heatmaps" in legacy_kwargs:
        attribution_maps = legacy_kwargs["gradcam_heatmaps"]
    if attribution_scores is None and "gradcam_scores" in legacy_kwargs:
        attribution_scores = legacy_kwargs["gradcam_scores"]
    if attribution_evaluator is None and "gradcam_evaluator" in legacy_kwargs:
        attribution_evaluator = legacy_kwargs["gradcam_evaluator"]

    if attribution_maps is None or attribution_scores is None:
        raise ValueError("attribution_maps and attribution_scores are required")

    if attribution_renderer is None:
        from pytorch_grad_cam.utils.image import show_cam_on_image

        def _default_renderer(frame_float, attribution_map):
            frame_h, frame_w = frame_float.shape[:2]
            attribution_map = np.asarray(attribution_map, dtype=np.float32)
            if attribution_map.ndim == 3 and attribution_map.shape[-1] == 1:
                attribution_map = attribution_map[..., 0]
            if attribution_map.shape[:2] != (frame_h, frame_w):
                attribution_map = cv2.resize(
                    attribution_map,
                    (frame_w, frame_h),
                    interpolation=cv2.INTER_LINEAR,
                )
            attribution_map = np.clip(attribution_map, 0.0, 1.0)
            return show_cam_on_image(frame_float, attribution_map, use_rgb=True)

        attribution_renderer = _default_renderer

    num_frames = lowres_frames.shape[0]
    target_size = 448

    output_path = os.path.join(video_directory, f"{video_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None

    for idx in range(num_frames):
        frame_raw = lowres_frames[idx]
        if hasattr(frame_raw, "cpu"):
            frame_raw = frame_raw.cpu().numpy()
        frame_raw = frame_raw.astype(np.uint8)

        frame_float = _normalize_frame_float(frame_raw)
        attribution_vis = attribution_renderer(frame_float, attribution_maps[idx])
        attribution_vis_resized = cv2.resize(attribution_vis, (target_size, target_size))

        if attribution_evaluator is not None and matched_video_id is not None:
            annots = attribution_evaluator.get_annotations_for_frame(matched_video_id, idx)
            if annots:
                roi_annots = [a for a in annots if not a.get("no_roi", False)]
                if roi_annots:
                    frame_metrics = attribution_evaluator.evaluate_frame(
                        attribution_maps[idx], matched_video_id, idx,
                        cam_threshold=cam_threshold,
                    )
                    if draw_overlay_bboxes:
                        from coco_attribution_eval import draw_gt_bboxes_on_frame

                        attribution_vis_resized = draw_gt_bboxes_on_frame(
                            attribution_vis_resized,
                            roi_annots,
                            attribution_evaluator.categories,
                            thickness=4,
                        )

        panel_width = target_size

        if render_text:
            fig, ax = plt.subplots(figsize=(panel_width / 100, 2.0))
            ax.plot(np.arange(num_frames), attribution_scores,
                    marker="o", linewidth=2, markersize=4, color="blue")
            ax.plot(idx, attribution_scores[idx],
                    marker="o", markersize=10, color="red", zorder=5)
            ax.set_xlabel("Frame")
            ax.set_ylabel(attribution_method_name)
            ax.set_title(f"{attribution_method_name} Scores - {ground_truth_text}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            graph_img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            graph_img = graph_img.reshape(canvas.get_width_height()[::-1] + (4,))
            graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2RGB)
            plt.close(fig)
        else:
            graph_img = _render_text_free_graph(attribution_scores, idx, panel_width)
        graph_img = cv2.resize(graph_img, (panel_width, graph_img.shape[0]))

        combined = np.vstack([attribution_vis_resized, graph_img])

        if out is None:
            h, w = combined.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, 1, (w, h))

        out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    if out is not None:
        out.release()
        print(f"Low-res visualization saved to: {output_path}")
    else:
        print("Error: Could not create low-res visualization video.")
