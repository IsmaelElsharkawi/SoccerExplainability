import json
import os
import cv2

COCO_PATH = "annotations-coco.json"
JSON_PATH = "train_data/json/one_video.json"

# BGR colors per category tier (no green, blue, yellow, or red/magenta)
TIER_COLORS = {
    "small label": (255, 0, 150),   # vivid purple RGB(150,0,255)
    "large label": (147, 20, 255),  # deep pink   RGB(255,20,147)
    "visual cue":  (220, 182, 255),  # light pink BGR → RGB(255,182,220)
}
BOX_THICKNESS = 5
VIDEOS_BASE = (
    "/mnt/c/Users/z004kjmt/Downloads"
    "/SoccerLensVisualizations-no-bboxes-20260421T231816Z-3-001_v2"
    "/SoccerLensVisualizations-no-bboxes"
)
ORIGINAL_VIDEOS_BASE = (
    "/mnt/c/Users/z004kjmt/Downloads"
    "/SoccerLens-for-annotation/SoccerLens-for-annotation"
)
OUTPUT_DIR = "extracted_frames"

MODELS = ["MatchVision", "SigLip", "SoccerMaster"]
METHODS = ["Chefer", "GradCam"]

METHOD_TAG = {
    "Chefer": "chefer",
    "GradCam": "gradcam",
}

MODEL_TAG = {
    "MatchVision": "MatchVision",
    "SigLip": "SigLip",
    "SoccerMaster": "SoccerMaster",
}


def extract_frame(video_path, time_sec, crop=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: cannot open {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"  ERROR: invalid fps for {video_path}")
        cap.release()
        return None
    frame_idx = int(time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  ERROR: cannot read frame {frame_idx} from {video_path}")
        return None
    if crop:
        h, w = frame.shape[:2]
        frame = frame[0:w, 0:w]
    return frame


def load_coco_index(coco_path):
    """Return (ann_index, cat_map, img_size_map) where:
    ann_index[(video_id, second)] -> list of annotations
    cat_map[category_id] -> category name
    img_size_map[(video_id, second)] -> (width, height)
    """
    with open(coco_path, "r") as f:
        coco = json.load(f)
    cat_map = {c["id"]: c["name"] for c in coco.get("categories", [])}
    ann_index = {}
    img_size_map = {}
    for img in coco.get("images", []):
        last_us = img["id"].rfind("_")
        video_id = img["id"][:last_us]
        try:
            second = int(img["id"][last_us + 1:])
        except ValueError:
            continue
        key = (video_id, second)
        ann_index.setdefault(key, [])
        img_size_map[key] = (img["width"], img["height"])
    for ann in coco.get("annotations", []):
        if ann.get("no_roi"):
            continue
        # image_id format: "{video_id}_{second}"
        img_id = ann["image_id"]
        last_us = img_id.rfind("_")
        video_id = img_id[:last_us]
        try:
            second = int(img_id[last_us + 1:])
        except ValueError:
            continue
        key = (video_id, second)
        ann_index.setdefault(key, []).append(ann)
    return ann_index, cat_map, img_size_map


def draw_bboxes(frame, ann_index, cat_map, img_size_map, video_id, vis_time):
    """Draw color-coded bounding boxes onto frame (returns a copy)."""
    second = int(round(vis_time))
    anns = ann_index.get((video_id, second), [])
    if not anns:
        return frame
    out = frame.copy()
    fh, fw = out.shape[:2]
    ref_w, ref_h = img_size_map.get((video_id, second), (fw, fh))
    for ann in anns:
        cat_name = cat_map.get(ann["category_id"], "unknown")
        color = TIER_COLORS.get(cat_name, (255, 255, 0))
        x, y, bw, bh = ann["bbox"]
        # Detect normalized coords: all values in [0, 2] range
        # Always scale against actual frame size so placeholder (1,1) doesn't break things
        if max(x, y, bw, bh) <= 2.0:
            x *= fw; y *= fh; bw *= fw; bh *= fh
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + bw), int(y + bh)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, BOX_THICKNESS)
    return out


def main():
    ann_index, cat_map, img_size_map = load_coco_index(COCO_PATH)

    with open(JSON_PATH, "r") as f:
        entries = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for entry in entries:
        video_rel = entry["video"]
        caption = entry["caption"]
        vis_time = entry["visualization_time"]

        # The video files are stored under <match>/<clip>.mp4
        # JSON path: league/match/clip.mp4 -> match key is match/clip.mp4
        parts = video_rel.replace("\\", "/").split("/")
        match_clip = "/".join(parts[1:])  # e.g. "2016-12-04 - .../2_20_48.mp4"

        caption_tag = caption.replace(" ", "_")

        # Extract original frame with bounding boxes
        orig_video_path = os.path.join(ORIGINAL_VIDEOS_BASE, video_rel.replace("\\", "/"))
        orig_out_path = os.path.join(OUTPUT_DIR, f"{caption_tag}_original.png")
        if os.path.exists(orig_video_path):
            orig_frame = extract_frame(orig_video_path, vis_time, crop=False)
            if orig_frame is not None:
                orig_frame = draw_bboxes(orig_frame, ann_index, cat_map, img_size_map, video_rel.replace("\\", "/"), vis_time)
                cv2.imwrite(orig_out_path, orig_frame)
                print(f"  Saved: {orig_out_path}")
        else:
            print(f"  MISSING original: {orig_video_path}")

        for model in MODELS:
            for method in METHODS:
                video_path = os.path.join(VIDEOS_BASE, model, method, match_clip)
                if not os.path.exists(video_path):
                    print(f"  MISSING: {video_path}")
                    continue

                frame = extract_frame(video_path, vis_time)
                if frame is None:
                    continue

                out_name = f"{caption_tag}_{METHOD_TAG[method]}_{MODEL_TAG[model]}.png"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                cv2.imwrite(out_path, frame)
                print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
