import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from coco_attribution_eval import draw_gt_bboxes_on_frame, overlay_metrics_text


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

    text_height = 80
    text_area = np.ones((text_height, hr_w + right_column.shape[1], 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    cv2.putText(text_area, f"Frame: {frame_number}/30", (10, 25),
                font, font_scale, (0, 0, 0), thickness)

    gt_color = (0, 150, 0) if prediction_text == ground_truth_text else (0, 0, 200)
    cv2.putText(text_area, f"Ground Truth: {ground_truth_text}", (10, 50),
                font, font_scale, gt_color, thickness)

    pred_color = (0, 150, 0) if prediction_text == ground_truth_text else (0, 0, 200)
    cv2.putText(text_area, f"Prediction: {prediction_text}", (10, 75),
                font, font_scale, pred_color, thickness)

    combined_main = np.hstack([high_res_frame, right_column])
    final_frame = np.vstack([text_area, combined_main])
    return final_frame


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
        plt.plot(frame_numbers, attribution_scores, marker="o", linewidth=2, markersize=4, color="blue")
        plt.plot(idx, attribution_scores[idx], marker="o", markersize=10, color="red", zorder=5)
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

        text_height = 80
        text_area = np.ones((text_height, hr_w + right_column.shape[1], 3), dtype=np.uint8) * 255

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        cv2.putText(text_area, f"Frame: {current_attribution_idx + 1}/30", (10, 25),
                    font, font_scale, (0, 0, 0), thickness)

        gt_color = (0, 150, 0) if prediction_text == ground_truth_text else (0, 0, 200)
        cv2.putText(text_area, f"Ground Truth: {ground_truth_text}", (10, 50),
                    font, font_scale, gt_color, thickness)

        pred_color = (0, 150, 0) if prediction_text == ground_truth_text else (0, 0, 200)
        cv2.putText(text_area, f"Prediction: {prediction_text}", (10, 75),
                    font, font_scale, pred_color, thickness)

        combined_main = np.hstack([high_res_frame, right_column])
        combined_frame = np.vstack([text_area, combined_main])

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
                                    attribution_renderer=None, **legacy_kwargs):
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

        frame_float = np.float32(cv2.resize(frame_raw, (224, 224))) / 255.0
        attribution_vis = attribution_renderer(frame_float, attribution_maps[idx])
        attribution_vis_resized = cv2.resize(attribution_vis, (target_size, target_size))

        if attribution_evaluator is not None and matched_video_id is not None:
            annots = attribution_evaluator.get_annotations_for_frame(matched_video_id, idx)
            if annots:
                roi_annots = [a for a in annots if not a.get("no_roi", False)]
                if roi_annots:
                    attribution_vis_resized = draw_gt_bboxes_on_frame(
                        attribution_vis_resized,
                        roi_annots,
                        attribution_evaluator.categories,
                        thickness=3,
                    )

                    frame_metrics = attribution_evaluator.evaluate_frame(
                        attribution_maps[idx], matched_video_id, idx,
                        cam_threshold=cam_threshold,
                    )
                    if frame_metrics:
                        attribution_vis_resized = overlay_metrics_text(
                            attribution_vis_resized, frame_metrics)

        panel_width = target_size

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
        graph_img = cv2.resize(graph_img, (panel_width, graph_img.shape[0]))

        text_height = 80
        text_bar = np.ones((text_height, panel_width, 3), dtype=np.uint8) * 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        match = prediction_text == ground_truth_text
        color = (0, 220, 0) if match else (0, 0, 220)
        cv2.putText(text_bar, f"Frame {idx + 1}/{num_frames}",
                    (10, 25), font, 0.55, (255, 255, 255), 1)
        cv2.putText(text_bar, f"GT: {ground_truth_text}",
                    (10, 50), font, 0.55, color, 2)
        cv2.putText(text_bar, f"Pred: {prediction_text}",
                    (10, 72), font, 0.55, color, 2)

        combined = np.vstack([text_bar, attribution_vis_resized, graph_img])

        if out is None:
            h, w = combined.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, 1, (w, h))

        out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    if out is not None:
        out.release()
        print(f"Low-res visualization saved to: {output_path}")
    else:
        print("Error: Could not create low-res visualization video.")
