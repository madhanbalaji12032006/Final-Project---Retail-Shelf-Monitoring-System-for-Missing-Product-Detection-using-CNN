import argparse, os, yaml, csv, json
from pathlib import Path
import torch, cv2
from PIL import Image
from torchvision import transforms as T
from coco_utils import load_categories_from_coco
from utils.viz import draw_detections

def load_model(ckpt_path, num_classes, device):
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-folder", required=True, help="folder of images to run inference on")
    ap.add_argument("--annotations", required=True, help="COCO annotations (to get class names)")
    ap.add_argument("--planogram", required=True, help="YAML mapping: class name -> expected count")
    ap.add_argument("--checkpoint", required=True, help="trained model checkpoint (.pt)")
    ap.add_argument("--output-dir", default="artifacts/outputs")
    ap.add_argument("--score-thresh", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # class names from COCO
    cat_id_to_name, _, class_names = load_categories_from_coco(args.annotations)
    # map contiguous label ids (1..N) used in training to names in id order
    # We assume categories were contiguous-sorted when training (CocoDetDataset does that).
    label_map = {i+1: name for i, name in enumerate(sorted(cat_id_to_name.values()))}

    num_classes = len(label_map) + 1  # + background

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, num_classes, device)

    # load planogram
    with open(args.planogram, "r", encoding="utf-8") as f:
        planogram = yaml.safe_load(f) or {}

    # inference loop
    tfm = T.Compose([T.ToTensor()])

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(sorted(Path(args.images_folder).glob(ext)))

    per_class_counts = {k: 0 for k in planogram.keys()}
    per_image_counts = {}

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        tensor = tfm(img).to(device)
        with torch.no_grad():
            out = model([tensor])[0]

        # convert to cpu numpy for viz
        boxes = out["boxes"].cpu().numpy()
        labels = out["labels"].cpu().numpy().tolist()
        scores = out["scores"].cpu().numpy()

        # count by class name with score filter
        counts = {}
        for lbl, sc in zip(labels, scores):
            if sc < args.score_thresh:
                continue
            name = label_map.get(int(lbl), str(lbl))
            counts[name] = counts.get(name, 0) + 1

        per_image_counts[img_path.name] = counts
        # update global counts only for planogram classes
        for name, c in counts.items():
            if name in per_class_counts:
                per_class_counts[name] += c

        # draw & save annotated image
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        ann = draw_detections(bgr, out["boxes"].cpu().numpy(), out["labels"].cpu().numpy(), out["scores"].cpu().numpy(), label_map, args.score_thresh)
        save_path = Path(args.output_dir) / f"ann_{img_path.stem}.jpg"
        cv2.imwrite(str(save_path), ann)

    # compute missing vs planogram
    missing = {}
    for name, expected in planogram.items():
        detected = per_class_counts.get(name, 0)
        delta = int(expected) - int(detected)
        if delta > 0:
            missing[name] = delta

    # save reports
    csv_path = Path(args.output_dir) / "missing_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Expected", "Detected", "Missing"])
        for name, expected in planogram.items():
            det = per_class_counts.get(name, 0)
            writer.writerow([name, expected, det, max(0, int(expected) - int(det))])

    json_path = Path(args.output_dir) / "missing_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "per_image_counts": per_image_counts,
            "aggregate_detected": per_class_counts,
            "planogram": planogram,
            "missing": missing
        }, f, indent=2)

    print(f"Saved outputs to {args.output_dir}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")

if __name__ == "__main__":
    import numpy as np  # needed for viz
    main()
