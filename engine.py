import math, time, torch
from torch.utils.data import DataLoader
from collections import defaultdict

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50, scaler=None):
    model.train()
    header = f"Epoch [{epoch}]"
    loss_hist = []

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                losses_dict = model(images, targets)
                loss = sum(loss for loss in losses_dict.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses_dict = model(images, targets)
            loss = sum(loss for loss in losses_dict.values())
            loss.backward()
            optimizer.step()

        loss_value = loss.item()
        loss_hist.append(loss_value)

        if (i + 1) % print_freq == 0:
            print(f"{header} | Iter {i+1}/{len(data_loader)} | Loss: {loss_value:.4f}")

    return float(sum(loss_hist) / max(1, len(loss_hist)))

@torch.no_grad()
def evaluate(model, data_loader, device, iou_thresh=0.5, score_thresh=0.5):
    """
    Very light eval: compute per-class true positives by naive IOU matching.
    For quick model selection; for full metrics use pycocotools COCOeval.
    """
    from torchvision.ops import box_iou

    model.eval()
    total = defaultdict(lambda: {"tp":0, "pred":0, "gt":0})

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            # filter scores
            keep = out["scores"] >= score_thresh
            pred_boxes = out["boxes"][keep].cpu()
            pred_labels = out["labels"][keep].cpu()

            gt_boxes = tgt["boxes"]
            gt_labels = tgt["labels"]

            if len(pred_boxes) == 0:
                # count all gt as missed
                for l in gt_labels.tolist():
                    total[l]["gt"] += 1
                continue

            ious = box_iou(pred_boxes, gt_boxes)
            used_gt = set()
            for pi, pl in enumerate(pred_labels.tolist()):
                total[pl]["pred"] += 1
                # match best gt of same class
                gi = torch.argmax(ious[pi])
                if ious[pi, gi] >= iou_thresh and gi.item() not in used_gt and pl == gt_labels[gi].item():
                    total[pl]["tp"] += 1
                    used_gt.add(gi.item())

            for l in gt_labels.tolist():
                total[l]["gt"] += 1

    # compute precision/recall per class
    metrics = {}
    for cls_id, v in total.items():
        tp, pred, gt = v["tp"], v["pred"], v["gt"]
        prec = tp / pred if pred else 0.0
        rec = tp / gt if gt else 0.0
        metrics[cls_id] = {"precision": prec, "recall": rec, "tp": tp, "pred": pred, "gt": gt}

    return metrics
