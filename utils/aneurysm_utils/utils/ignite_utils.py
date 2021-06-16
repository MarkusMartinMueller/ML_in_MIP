from collections.abc import Mapping
import numpy as np
from ignite.metrics import Metric
import torch
from typing import Optional, Sequence, Tuple, Union
from ignite.utils import convert_tensor


def prepare_batch(
    batch: Sequence[torch.Tensor],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
) -> Tuple[Union[torch.Tensor, Sequence, Mapping, str, bytes], ...]:
    """Prepare batch for training: pass to a device with options."""
    try:
        x, y = batch
        return (
            convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking),
        )
    except ValueError:
        batch_tensor, pos, ptr, x, y = batch
        batch_tensor = convert_tensor(
            batch_tensor, device=device, non_blocking=non_blocking
        )[1]
        pos = convert_tensor(pos, device=device, non_blocking=non_blocking)[1]
        x = convert_tensor(x, device=device, non_blocking=non_blocking)[1]
        y = convert_tensor(y, device=device, non_blocking=non_blocking)[1]
        return {"batch": batch_tensor, "pos": pos, "x": x}, y


class IntersectionOverUnion(Metric):
    """Computes the intersection over union (IoU) per class.
    based on: https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
    - `update` must receive output of the form `(y_pred, y)`.
    """

    def __init__(self, num_classes=10, ignore_index=255, output_transform=lambda x: x):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes))

        super(IntersectionOverUnion, self).__init__(output_transform=output_transform)

    def _fast_hist(self, label_true, label_pred):
        # mask = (label_true >= 0) & (label_true < self.num_classes)
        mask = label_true != self.ignore_index
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(np.int) + label_pred[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, output):
        y_pred, y = output

        for label_true, label_pred in zip(y.numpy(), y_pred.numpy()):
            self.confusion_matrix += self._fast_hist(
                label_true.flatten(), label_pred.flatten()
            )

    def compute(self):
        hist = self.confusion_matrix
        with np.errstate(divide="ignore", invalid="ignore"):
            iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

        return np.nanmean(iu)


class MeanAveragePrecision(Metric):
    def __init__(self, num_classes=20, output_transform=lambda x: x):
        super(MeanAveragePrecision, self).__init__(output_transform=output_transform)

        self.num_classes = num_classes

    def reset(self):
        self._true_boxes = torch.tensor([], dtype=torch.long)
        self._true_labels = torch.tensor([], dtype=torch.long)

        self._det_boxes = torch.tensor([], dtype=torch.float32)
        self._det_labels = torch.tensor([], dtype=torch.float32)
        self._det_scores = torch.tensor([], dtype=torch.float32)

    def update(self, output):
        boxes_preds, labels_preds, scores_preds, boxes, labels = output

        self._true_boxes = torch.cat([self._true_boxes, boxes], dim=0)
        self._true_labels = torch.cat([self._true_labels, labels], dim=0)

        self._det_boxes = torch.cat([self._det_boxes, boxes_preds], dim=0)
        self._det_labels = torch.cat([self._det_labels, labels_preds], dim=0)
        self._det_scores = torch.cat([self._det_scores, scores_preds], dim=0)

    def compute(self):
        for c in range(1, self.num_classes):
            pass
