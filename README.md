# Retail Shelf Monitoring System for Missing Product Detection using CNN

Detect missing products on retail shelves by training an object detector (Faster R-CNN with a ResNet-50 backbone) and comparing detections to a store planogram (expected product counts per class).

> **Why this approach?** Faster R-CNN is a CNN-based detector (feature pyramid + ResNet-50) that performs well for product/localization tasks when training data is limited, and it’s built into `torchvision` for a clean training loop.

---

## Features
- **CNN-based** object detector: `torchvision.models.detection.fasterrcnn_resnet50_fpn`.
- **COCO-style annotations** for training.
- **Planogram-driven missing detection**: YAML file with expected counts per product class.
- **Batch inference** on a folder of shelf images, annotated outputs, and a CSV/JSON report of missing items.
- Clean, minimal training/eval loop with checkpoints.

---

## Repo Structure
```
retail-shelf-missing-products/
├── README.md
├── requirements.txt
├── src/
│   ├── train.py
│   ├── infer.py
│   ├── coco_utils.py
│   └── utils/
│       ├── engine.py
│       ├── transforms.py
│       └── viz.py
├── data/
│   ├── train/
│   │   ├── images/                # your training images
│   │   └── annotations.json       # COCO-style bboxes
│   └── val/
│       ├── images/
│       └── annotations.json
├── artifacts/
│   ├── checkpoints/               # saved models
│   └── outputs/                   # inference outputs
└── planogram.yaml                 # expected counts per class
```
> You can rename `data/` paths; pass them via CLI flags.

---

## Install
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Prepare Data (COCO format)
- Place images under `data/train/images` and `data/val/images`.
- Provide `annotations.json` matching COCO detection format (bbox = [x, y, width, height]).
- Categories **must** include the product class names used in the planogram.

If you are new to COCO, tools like [Label Studio], [Roboflow], or [CVAT] can export COCO detection annotations.

---

## Training
```bash
python src/train.py \
  --train-images data/train/images \
  --train-anno   data/train/annotations.json \
  --val-images   data/val/images \
  --val-anno     data/val/annotations.json \
  --epochs 20 --batch-size 4 --lr 0.0005 \
  --checkpoint-dir artifacts/checkpoints
```
- A model checkpoint (best mAP) is saved in `artifacts/checkpoints`.
- You can resume training with `--resume <path_to_ckpt.pt>`.

---

## Inference & Missing Product Report
First, define expected counts per class in `planogram.yaml`, for example:
```yaml
Apple Juice 1L: 8
Orange Juice 1L: 8
Cola 500ml: 12
Soda Water 1L: 6
```
Then run:
```bash
python src/infer.py \
  --images-folder data/val/images \
  --annotations   data/val/annotations.json \
  --planogram planogram.yaml \
  --checkpoint artifacts/checkpoints/best.pt \
  --output-dir artifacts/outputs \
  --score-thresh 0.5
```
Outputs:
- Annotated images with predicted boxes & labels.
- `missing_report.csv` and `missing_report.json` summarizing counts vs. planogram.

---

## Notes
- This repo uses **Faster R-CNN (CNN-based)**; if you prefer single-shot models (SSD/YOLO), you can swap the model while keeping the rest of the pipeline.
- Start with a moderate image size (e.g., 800px shorter side, which is default for Faster R-CNN).

---

## License
MIT
