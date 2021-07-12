import warnings
from typing import Callable, List, Optional, Sequence, Union
from collections.abc import Mapping
import numpy as np
import torch.nn as nn
from ignite.metrics import Metric
import torch
from typing import Optional, Sequence, Tuple, Union
from ignite.utils import convert_tensor
from torch.nn.modules.loss import _Loss
from monai.losses import DiceLoss

class DiceCELoss(_Loss):
    """
    Compute both Dice loss and Cross Entropy Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Cross Entropy Loss is shown in ``torch.nn.CrossEntropyLoss``. In this implementation,
    two deprecated parameters ``size_average`` and ``reduce``, and the parameter ``ignore_index`` are
    not supported.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> None:
        """
        Args:
            ``ce_weight`` and ``lambda_ce`` are only used for cross entropy loss.
            ``reduction`` is used for both losses and other parameters are only used for dice loss.

            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            ce_weight: a rescaling weight given to each class for cross entropy loss.
                See ``torch.nn.CrossEntropyLoss()`` for more information.
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_ce: the trade-off weight value for cross entropy loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=ce_weight,
            reduction=reduction,
        )
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """

        target = torch.unsqueeze(target, 1)
        if len(input.shape) != len(target.shape):
            raise ValueError("the number of dimensions for input and target should be the same.")

        dice_loss = self.dice(input, target)

        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch == n_target_ch:
            # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            target = torch.argmax(target, dim=1)
        else:
            target = torch.squeeze(target, dim=1)
        target = target.long()
        ce_loss = self.cross_entropy(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss
        return total_loss


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
