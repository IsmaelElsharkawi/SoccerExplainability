"""Write a COCO file that lists every frame of a clip list with no boxes.

The inference scripts only save saliency maps for clips that appear in the COCO
file passed as COCO_JSON. To run them on clips that have no human annotations
(for example the 800 extra clips), point COCO_JSON at this placeholder and score
afterwards against the real annotations with EVAL_COCO_JSON.

    python colab/make_placeholder_coco.py \
        train_data/json/selected_videos_for_annotations_800.json /content/placeholder_800-coco.json
"""
import json
import sys


def main(clip_list, out_path, num_frames=30, width=1280, height=720):
    clips = [c["video"] for c in json.load(open(clip_list))]
    images, annotations = [], []
    for video in clips:
        for second in range(num_frames):
            image_id = f"{video}_{second}"
            images.append({"id": image_id, "file_name": f"{video}_t{second}.jpg", "video_id": video,
                           "second": second, "width": width, "height": height})
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": None,
                                "bbox": [0, 0, 0, 0], "iscrowd": 0, "no_roi": True})
    coco = {"info": {"description": "placeholder: every frame of the clip list, no boxes"},
            "images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "small label"}, {"id": 2, "name": "large label"},
                           {"id": 3, "name": "visual cue"}]}
    json.dump(coco, open(out_path, "w"))
    print(f"{len(clips)} clips, {len(images)} frames -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
