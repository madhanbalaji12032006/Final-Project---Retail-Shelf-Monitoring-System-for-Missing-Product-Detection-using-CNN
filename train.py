import argparse, os, torch, yaml, json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_convert
import torchvision

from utils.engine import train_one_epoch, evaluate
from utils.transforms import Compose, ToTensor, RandomHorizontalFlip
from coco_utils import load_categories_from_coco

class CocoDetDataset(Dataset):
    def __init__(self, images_dir, annotations_path, transforms=None):
        from pycocotools.coco import COCO
        self.images_dir = Path(images_dir)
        self.coco = COCO(annotations_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

        # build category id mapping to contiguous ids (1..N)
        cat_ids = sorted(self.coco.getCatIds())
        self.catid_to_contig = {cid: i+1 for i, cid in enumerate(cat_ids)}
        self.contig_to_catid = {v: k for k, v in self.catid_to_contig.items()}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        from pycocotools.coco import maskUtils
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(self.images_dir / path).convert("RGB")

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for a in anns:
            if 'bbox' in a:
                # COCO bbox is [x, y, w, h] -> convert to [x1, y1, x2, y2]
                x, y, w, h = a['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(self.catid_to_contig[a['category_id']])
                areas.append(a.get('area', w*h))
                iscrowd.append(a.get('iscrowd', 0))

        import torch
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # replace the classifier head for our num_classes (including background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-images", required=True)
    ap.add_argument("--train-anno", required=True)
    ap.add_argument("--val-images", required=True)
    ap.add_argument("--val-anno", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--checkpoint-dir", default="artifacts/checkpoints")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true", help="enable mixed precision")
    args = ap.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Build datasets
    train_tfms = Compose([ToTensor(), RandomHorizontalFlip(0.5)])
    val_tfms   = Compose([ToTensor()])

    train_ds = CocoDetDataset(args.train_images, args.train_anno, transforms=train_tfms)
    val_ds   = CocoDetDataset(args.val_images, args.val_anno, transforms=val_tfms)

    # num classes = categories + background
    cat_id_to_name, _, _ = load_categories_from_coco(args.train_anno)
    num_classes = len(cat_id_to_name) + 1

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(num_classes).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    start_epoch = 1
    best_score = -1.0
    best_path = None

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 1)
        best_score = ckpt.get("best_score", -1.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best_score={best_score:.4f})")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        loss = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=20, scaler=scaler)
        print(f"Epoch {epoch} | avg loss: {loss:.4f}")

        # quick eval
        metrics = evaluate(model, val_loader, device, iou_thresh=0.5, score_thresh=0.5)
        mean_recall = sum(v["recall"] for v in metrics.values()) / max(1, len(metrics))
        print(f"Validation mean recall (0.5 IOU): {mean_recall:.4f}")

        # save checkpoint
        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_score": max(best_score, mean_recall),
        }
        save_path = Path(args.checkpoint_dir) / f"epoch_{epoch:03d}.pt"
        torch.save(ckpt, save_path)

        # track best
        if mean_recall > best_score:
            best_score = mean_recall
            best_path = Path(args.checkpoint_dir) / "best.pt"
            torch.save(ckpt, best_path)
            print(f"Saved new best checkpoint to {best_path} (mean_recall={best_score:.4f})")

    print("Training complete. Best:", best_path if best_path else "N/A")

if __name__ == "__main__":
    main()
