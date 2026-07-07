"""
Chỉ số đánh giá DÙNG CHUNG cho mọi model — đây là "1 protocol" mà thầy yêu cầu.

Chính (thầy chốt 04/07): IoU/mIoU, F1 (=Dice), Precision, Recall — theo lớp.
Phụ: pixel accuracy (vẫn báo, không làm chính vì mất cân bằng lớp).

Bỏ qua pixel nhãn -1. Hỗ trợ 2 cách tổng hợp flood-IoU:
  * gộp pixel toàn tập (micro)          -> compute()['flood_IoU']
  * trung bình theo chip CÓ flood       -> compute()['flood_IoU_perchip_mean']
Lớp: 0 nền · 1 nước thường trực · 2 nước lũ (flood_class=2).
"""
import numpy as np

CLASS_NAMES = ["background", "permanent_water", "flood"]


class SegMetrics:
    def __init__(self, num_classes=3, ignore_index=-1, flood_class=2):
        self.n = num_classes
        self.ignore = ignore_index
        self.flood = flood_class
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.flood_ious = []          # flood-IoU từng chip (chỉ chip có flood trong GT)
        self.perchip = []             # (chip_id, region, flood_iou) — cho Wilcoxon + per-region

    def update(self, pred, target):
        """Cộng dồn confusion matrix. pred/target: tensor int, shape (B,H,W) hoặc (H,W)."""
        pred = pred.detach().cpu().numpy().ravel()
        target = target.detach().cpu().numpy().ravel()
        mask = target != self.ignore
        t, p = target[mask], pred[mask]
        k = (t >= 0) & (t < self.n)
        self.cm += np.bincount(self.n * t[k] + p[k],
                               minlength=self.n ** 2).reshape(self.n, self.n)

    def update_perchip(self, pred, target, chip_id=None):
        """Tính flood-IoU cho 1 chip; chỉ lưu nếu GT chip đó CÓ flood."""
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        valid = target != self.ignore
        gt = (target == self.flood) & valid
        if gt.sum() == 0:
            return
        pr = (pred == self.flood) & valid
        inter = np.logical_and(gt, pr).sum()
        union = np.logical_or(gt, pr).sum()
        iou = inter / union if union > 0 else 0.0
        self.flood_ious.append(iou)
        if chip_id is not None:
            self.perchip.append((chip_id, str(chip_id).split("_")[0], float(iou)))

    def compute(self):
        cm = self.cm.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        iou = tp / (tp + fp + fn + 1e-9)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)       # F1 == Dice
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        out = {
            "IoU_per_class": dict(zip(CLASS_NAMES, iou.round(4).tolist())),
            "mIoU": float(iou.mean()),
            "F1_per_class": dict(zip(CLASS_NAMES, f1.round(4).tolist())),
            "precision_per_class": dict(zip(CLASS_NAMES, prec.round(4).tolist())),
            "recall_per_class": dict(zip(CLASS_NAMES, rec.round(4).tolist())),
            "pixel_accuracy": float(tp.sum() / (cm.sum() + 1e-9)),   # chỉ số PHỤ
            "flood_IoU": float(iou[self.flood]),                    # chỉ số CHÍNH
            "flood_F1": float(f1[self.flood]),
        }
        if self.flood_ious:
            out["flood_IoU_perchip_mean"] = float(np.mean(self.flood_ious))
            out["n_flood_chips"] = len(self.flood_ious)
        return out

    def reset(self):
        self.cm[:] = 0
        self.flood_ious.clear()
