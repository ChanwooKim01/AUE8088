# # YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
# """Loss functions."""

# import torch
# import torch.nn as nn

# from utils.metrics import bbox_iou
# from utils.torch_utils import de_parallel


# def smooth_BCE(eps=0.1):
#     """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441"""
#     return 1.0 - 0.5 * eps, 0.5 * eps


# class BCEBlurWithLogitsLoss(nn.Module):
#     # BCEwithLogitLoss() with reduced missing label effects.
#     def __init__(self, alpha=0.05):
#         """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
#         parameter.
#         """
#         super().__init__()
#         self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
#         self.alpha = alpha

#     def forward(self, pred, true):
#         """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
#         returns mean loss.
#         """
#         loss = self.loss_fcn(pred, true)
#         pred = torch.sigmoid(pred)  # prob from logits
#         dx = pred - true  # reduce only missing label effects
#         # dx = (pred - true).abs()  # reduce missing label and false label effects
#         alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
#         loss *= alpha_factor
#         return loss.mean()


# class FocalLoss(nn.Module):
#     # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
#     def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
#         """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
#         'none'.
#         """
#         super().__init__()
#         self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = loss_fcn.reduction
#         self.loss_fcn.reduction = "none"  # required to apply FL to each element

#     def forward(self, pred, true):
#         """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
#         loss = self.loss_fcn(pred, true)
#         # p_t = torch.exp(-loss)
#         # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

#         # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
#         pred_prob = torch.sigmoid(pred)  # prob from logits
#         p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
#         alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
#         modulating_factor = (1.0 - p_t) ** self.gamma
#         loss *= alpha_factor * modulating_factor

#         if self.reduction == "mean":
#             return loss.mean()
#         elif self.reduction == "sum":
#             return loss.sum()
#         else:  # 'none'
#             return loss


# class QFocalLoss(nn.Module):
#     # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
#     def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
#         """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
#         super().__init__()
#         self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = loss_fcn.reduction
#         self.loss_fcn.reduction = "none"  # required to apply FL to each element

#     def forward(self, pred, true):
#         """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
#         `gamma` and `alpha`.
#         """
#         loss = self.loss_fcn(pred, true)

#         pred_prob = torch.sigmoid(pred)  # prob from logits
#         alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
#         modulating_factor = torch.abs(true - pred_prob) ** self.gamma
#         loss *= alpha_factor * modulating_factor

#         if self.reduction == "mean":
#             return loss.mean()
#         elif self.reduction == "sum":
#             return loss.sum()
#         else:  # 'none'
#             return loss


# class ComputeLoss:
#     sort_obj_iou = False

#     # Compute losses
#     def __init__(self, model, autobalance=False):
#         """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
#         device = next(model.parameters()).device  # get model device
#         h = model.hyp  # hyperparameters

#         # Define criteria
#         BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
#         BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

#         # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
#         self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

#         # Focal loss
#         g = h["fl_gamma"]  # focal loss gamma
#         if g > 0:
#             BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

#         m = de_parallel(model).model[-1]  # Detect() module
#         self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
#         self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
#         self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
#         self.na = m.na  # number of anchors
#         self.nc = m.nc  # number of classes
#         self.nl = m.nl  # number of layers
#         self.anchors = m.anchors
#         self.device = device

#     def __call__(self, p, targets):  # predictions, targets
#         """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
#         lcls = torch.zeros(1, device=self.device)  # class loss
#         lbox = torch.zeros(1, device=self.device)  # box loss
#         lobj = torch.zeros(1, device=self.device)  # object loss
#         tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

#         # Losses
#         for i, pi in enumerate(p):  # layer index, layer predictions
#             b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
#             tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

#             n = b.shape[0]  # number of targets
#             if n:
#                 pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
#                 # pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

#                 # Regression
#                 pxy = pxy.sigmoid() * 2 - 0.5
#                 pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
#                 pbox = torch.cat((pxy, pwh), 1)  # predicted box
#                 iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
#                 lbox += (1.0 - iou).mean()  # iou loss

#                 # Objectness
#                 iou = iou.detach().clamp(0).type(tobj.dtype)
#                 if self.sort_obj_iou:
#                     j = iou.argsort()
#                     b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
#                 if self.gr < 1:
#                     iou = (1.0 - self.gr) + self.gr * iou

#                 # If prediction is matched (iou > 0.5) with bounding box marked as ignore,
#                 # do not calculate objectness loss
#                 ign_idx = (tcls[i] == -1) & (iou > self.hyp["iou_t"])
#                 keep = ~ign_idx
#                 b, a, gj, gi, iou = b[keep], a[keep], gj[keep], gi[keep], iou[keep]

#                 tobj[b, a, gj, gi] = iou  # iou ratio

#                 # Classification
#                 if self.nc > 1:  # cls loss (only if multiple classes)
#                     t = torch.full_like(pcls, self.cn, device=self.device)  # targets
#                     t[range(n), tcls[i]] = self.cp
#                     lcls += self.BCEcls(pcls, t)  # BCE

#                 # Append targets to text file
#                 # with open('targets.txt', 'a') as file:
#                 #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

#             obji = self.BCEobj(pi[..., 4], tobj)
#             lobj += obji * self.balance[i]  # obj loss
#             if self.autobalance:
#                 self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

#         if self.autobalance:
#             self.balance = [x / self.balance[self.ssi] for x in self.balance]
#         lbox *= self.hyp["box"]
#         lobj *= self.hyp["obj"]
#         lcls *= self.hyp["cls"]
#         bs = tobj.shape[0]  # batch size

#         return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

#     def build_targets(self, p, targets):
#         """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
#         indices, and anchors.
#         """
#         na, nt = self.na, targets.shape[0]  # number of anchors, targets
#         tcls, tbox, indices, anch = [], [], [], []
#         gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
#         ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
#         targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

#         g = 0.5  # bias
#         off = (
#             torch.tensor(
#                 [
#                     [0, 0],
#                     [1, 0],
#                     [0, 1],
#                     [-1, 0],
#                     [0, -1],  # j,k,l,m
#                     # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
#                 ],
#                 device=self.device,
#             ).float()
#             * g
#         )  # offsets

#         for i in range(self.nl):
#             anchors, shape = self.anchors[i], p[i].shape
#             gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

#             # Match targets to anchors
#             t = targets * gain  # shape(3,n,7)
#             if nt:
#                 # Matches
#                 r = t[..., 4:6] / anchors[:, None]  # wh ratio
#                 j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
#                 # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
#                 t = t[j]  # filter

#                 # Offsets
#                 gxy = t[:, 2:4]  # grid xy
#                 gxi = gain[[2, 3]] - gxy  # inverse
#                 j, k = ((gxy % 1 < g) & (gxy > 1)).T
#                 l, m = ((gxi % 1 < g) & (gxi > 1)).T
#                 j = torch.stack((torch.ones_like(j), j, k, l, m))
#                 t = t.repeat((5, 1, 1))[j]
#                 offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
#             else:
#                 t = targets[0]
#                 offsets = 0

#             # Define
#             bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
#             a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
#             gij = (gxy - offsets).long()
#             gi, gj = gij.T  # grid indices

#             # Append
#             indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
#             tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
#             anch.append(anchors[a])  # anchors
#             tcls.append(c)  # class

#         return tcls, tbox, indices, anch

# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Loss functions."""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441"""
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
        parameter.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
        returns mean loss.
        """
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # 이 부분에서 영감 얻음
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        self.center_weight = h.get("center_weight", 0.05)

    def __call__(self, p, targets):  # predictions, targets
        """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
        # -- Person-only filter --
        # drop any target whose class != 0
        if targets.numel():
            targets = targets[targets[:, 1] == 0]
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        lcenter = torch.zeros(1, device=self.device)  # NEW center-regression L1 loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # custom
                # center based loss: L1 loss
                # IoU 기반의 regression loss 외에, center regression을 위한 L1 loss 추가
                # 기존 IoU 기반 regression 방법의 한계점 극복을 위함
                txy = tbox[i][:, :2]
                lcenter += nn.L1Loss()(pxy, txy)                  

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou

                # If prediction is matched (iou > 0.5) with bounding box marked as ignore,
                # do not calculate objectness loss                
                ign_idx = (tcls[i] == -1) & (iou > self.hyp["iou_t"])
                keep = ~ign_idx
                # 매칭된 앵커가 하나도 없으면 이 레이어 건너뛰기
                # b, a, gj, gi, iou = b[keep], a[keep], gj[keep], gi[keep], iou[keep]
                # Augmentation 등을 특히 수행했을 때, 모델 학습 시 배치에서 anchor와 매칭되는 값이 단 하나도 존재하지 않을 시 이 부분에서 오류 발생
                # 아마, 데이터셋 자체에 label이 없는 이미지가 너무 많아서 자주 터지는 듯
                if (iou.numel() == 0 or keep.numel() == 0 or iou.dim() == 0 or keep.sum() == 0 or
                    b.numel() == 0 or a.numel() == 0 or gj.numel() == 0 or gi.numel() == 0):
                    continue
                # 추가 안전 검사: tensor 차원 및 크기 일치 확인
                try:
                    if len(b) != len(keep) or len(iou) != len(keep):
                        continue
                    keep_indices = keep.nonzero().squeeze(-1)
                    if keep_indices.numel() == 0:
                        continue
                    b, a, gj, gi, iou = b[keep], a[keep], gj[keep], gi[keep], iou[keep]
                except (IndexError, RuntimeError) as e:
                    print(f"Warning: Skipping batch due to indexing error: {e}")
                    continue
                tobj[b, a, gj, gi] = iou  # iou ratio

                if self.nc > 1:  # cls loss (only if multiple classes)
                    person_exp_confidence = pcls.select(dim=-1, index=0)
                    person_mask = (tcls[i] == 0)  # person class index is 0
                    non_person_mask = ~person_mask

                    person_indices = []
                    non_person_indices = []
                    for idx in range(len(person_mask)):
                        if person_mask[idx] == True:
                            person_indices.append(idx)
                        else:
                            non_person_indices.append(idx)
                    
                    # Person인 경우 처리                             
                    if len(person_indices) > 0:
                        person_predictions = []
                        person_targets = []
                        for idx in person_indices:
                            person_predictions.append(person_exp_confidence[idx])
                            person_targets.append(self.cp)
                        
                        person_pred_tensor = torch.stack(person_predictions)
                        person_target_tensor = torch.tensor(person_targets, device=self.device)
                        lcls += self.BCEcls(person_pred_tensor, person_target_tensor)

                    # Non-person인 경우 처리
                    if len(non_person_indices) > 0:
                        non_person_predictions = []
                        non_person_targets = []
                        for idx in non_person_indices:
                            non_person_predictions.append(person_exp_confidence[idx])
                            non_person_targets.append(self.cn)
                        
                        non_person_pred_tensor = torch.stack(non_person_predictions)
                        non_person_target_tensor = torch.tensor(non_person_targets, device=self.device)
                        lcls += self.BCEcls(non_person_pred_tensor, non_person_target_tensor)

                    # t[range(n), tcls[i]] = self.cp
                    # lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        lcenter *= self.center_weight                           # scale center loss
        lbox += lcenter                     # include center loss in total
        bs = tobj.shape[0]  # batch size

        # include center loss in total
        loss = (lbox + lobj + lcls + lcenter) * bs
        return loss, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

# class ComputeLoss:
#     sort_obj_iou = False

#     # Compute losses
#     def __init__(self, model, is_train=True, autobalance=False):
#         """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
#         self.is_train = is_train
#         device = next(model.parameters()).device  # get model device
#         h = model.hyp  # hyperparameters

#         # Define criteria
#         BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
#         BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

#         # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
#         self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

#         # Focal loss
#         g = h["fl_gamma"]  # focal loss gamma
#         if g > 0:
#             BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

#         m = de_parallel(model).model[-1]  # Detect() module
#         self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
#         self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
#         self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
#         self.na = m.na  # number of anchors
#         self.nc = m.nc  # number of classes
#         self.nl = m.nl  # number of layers
#         self.anchors = m.anchors
#         self.device = device
        
#         # MultiBoxLoss style parameters
#         self.neg_pos_ratio = h.get("neg_pos_ratio", 3)  # Hard negative mining ratio
#         self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')  # For multi-label loss
        
#     def _to_one_hot(self, labels, n_dims):
#         """Convert labels to one-hot encoding"""
#         one_hot = torch.zeros(labels.size(0), n_dims, device=self.device)
#         one_hot.scatter_(1, labels.unsqueeze(1), 1)
#         return one_hot
        
#     def __call__(self, p, targets):  # predictions, targets
#         """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
#         if not self.is_train:
#             return 0.0
        
#         lcls = torch.zeros(1, device=self.device)  # class loss
#         lbox = torch.zeros(1, device=self.device)  # box loss
#         lobj = torch.zeros(1, device=self.device)  # object loss
#         lcenter = torch.zeros(1, device=self.device)  # center loss
#         tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        
#         # Collect all positive priors across layers for hard negative mining
#         all_cls_losses = []
#         all_positive_masks = []
#         all_cls_predictions = []
#         all_cls_targets = []
#         total_positives = 0
        
#         # Losses
#         for i, pi in enumerate(p):  # layer index, layer predictions
#             b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
#             tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

#             n = b.shape[0]  # number of targets
#             if n:
#                 pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0

#                 # Regression
#                 pxy = pxy.sigmoid() * 2 - 0.5
#                 pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
#                 pbox = torch.cat((pxy, pwh), 1)  # predicted box
                
#                 txy = tbox[i][:, :2] 
#                 lcenter += nn.L1Loss()(pxy, txy)
#                 iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
#                 lbox += (1.0 - iou).mean()  # iou loss

#                 # Objectness
#                 iou = iou.detach().clamp(0).type(tobj.dtype)
#                 if self.sort_obj_iou:
#                     j = iou.argsort()
#                     b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
#                 if self.gr < 1:
#                     iou = (1.0 - self.gr) + self.gr * iou

#                 # If prediction is matched (iou > 0.5) with bounding box marked as ignore,
#                 # do not calculate objectness loss
#                 ign_idx = (tcls[i] == -1) & (iou > self.hyp["iou_t"])
#                 keep = ~ign_idx
                
#                 # Enhanced defensive logic
#                 if (iou.numel() == 0 or keep.numel() == 0 or iou.dim() == 0 or keep.sum() == 0 or
#                     b.numel() == 0 or a.numel() == 0 or gj.numel() == 0 or gi.numel() == 0):
#                     continue
                    
#                 try:
#                     if len(b) != len(keep) or len(iou) != len(keep):
#                         continue
#                     keep_indices = keep.nonzero().squeeze(-1)
#                     if keep_indices.numel() == 0:
#                         continue
#                     b, a, gj, gi, iou = b[keep], a[keep], gj[keep], gi[keep], iou[keep]
#                 except (IndexError, RuntimeError) as e:
#                     print(f"Warning: Skipping batch due to indexing error: {e}")
#                     continue

#                 tobj[b, a, gj, gi] = iou  # iou ratio
                
#                 # === MultiBoxLoss Style Classification ===
#                 if self.nc > 1:  # cls loss (only if multiple classes)
#                     # Get all predictions for this layer
#                     layer_cls_pred = pi[..., 5:].view(-1, self.nc)  # All predictions (batch_size * H * W * anchors, n_classes)
                    
#                     # Create targets for all anchors in this layer
#                     batch_size = pi.shape[0]
#                     n_anchors_total = pi.shape[1] * pi.shape[2] * pi.shape[3]  # H * W * anchors
                    
#                     # Initialize all targets as background (class 0)
#                     layer_cls_targets = torch.zeros((batch_size * n_anchors_total), dtype=torch.long, device=self.device)
                    
#                     # Set positive targets
#                     if len(b) > 0:  # if there are positive samples
#                         flat_indices = b * n_anchors_total + a * pi.shape[2] * pi.shape[3] + gj * pi.shape[3] + gi
#                         layer_cls_targets[flat_indices] = tcls[i][keep]
#                         total_positives += len(b)
                    
#                     # Handle ignore indices (-1 labels)
                    
#                     ignore_mask = layer_cls_targets == -1
                    
#                     # Handle "People" class (label 3) - MultiBoxLoss style multi-label
#                     pair_indices = torch.nonzero((layer_cls_targets == 3), as_tuple=False).squeeze(-1)
                    
#                     # Convert to one-hot with multi-label support
#                     if self.nc >= 4:  # Ensure we have enough classes
#                         layer_cls_targets_adjusted = layer_cls_targets.clone()
#                         layer_cls_targets_adjusted[layer_cls_targets_adjusted == -1] = 0  # Set ignore to background temporarily
                        
#                         # Create one-hot encoding (adding 1 to make room for background)
#                         cls_targets_onehot = self._to_one_hot(layer_cls_targets_adjusted + 1, n_dims=self.nc + 1)[:, 1:self.nc+1]
                        
#                         # Set "People" class to activate both Person and Cyclist (assuming classes 1 and 2)
#                         if len(pair_indices) > 0 and self.nc >= 3:
#                             cls_targets_onehot[pair_indices, 1] = 1  # Person
#                             cls_targets_onehot[pair_indices, 2] = 1  # Cyclist
#                     else:
#                         # Fallback for insufficient classes
#                         cls_targets_onehot = torch.zeros((batch_size * n_anchors_total, self.nc), device=self.device)
#                         valid_mask = (layer_cls_targets >= 0) & (layer_cls_targets < self.nc)
#                         if valid_mask.sum() > 0:
#                             cls_targets_onehot[valid_mask, layer_cls_targets[valid_mask]] = 1
                    
#                     # Calculate loss for all predictions
#                     cls_loss_all = self.loss_fn(layer_cls_pred, cls_targets_onehot).sum(dim=1)
                    
#                     # Set ignore indices to zero
#                     cls_loss_all[ignore_mask] = 0.0
                    
#                     # Store for hard negative mining
#                     positive_mask = layer_cls_targets > 0
#                     all_cls_losses.append(cls_loss_all)
#                     all_positive_masks.append(positive_mask)
#                     all_cls_predictions.append(layer_cls_pred)
#                     all_cls_targets.append(cls_targets_onehot)

#             obji = self.BCEobj(pi[..., 4], tobj)
#             lobj += obji * self.balance[i]  # obj loss
#             if self.autobalance:
#                 self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

#         # === MultiBoxLoss Style Hard Negative Mining ===
#         if len(all_cls_losses) > 0 and total_positives > 0:
#             # Concatenate all losses
#             all_losses = torch.cat(all_cls_losses)
#             all_pos_masks = torch.cat(all_positive_masks)
            
#             # Positive classification loss
#             cls_loss_pos = all_losses[all_pos_masks]
            
#             # Hard negative mining
#             cls_loss_neg = all_losses.clone()
#             cls_loss_neg[all_pos_masks] = 0.0  # Remove positive samples
            
#             # Sort negatives by loss (hardest first)
#             neg_losses_sorted, neg_indices = cls_loss_neg.sort(descending=True)
            
#             # Select top hard negatives
#             n_hard_negatives = min(self.neg_pos_ratio * total_positives, (cls_loss_neg > 0).sum().item())
#             if n_hard_negatives > 0:
#                 cls_loss_hard_neg = neg_losses_sorted[:n_hard_negatives]
                
#                 # Final classification loss (MultiBoxLoss style)
#                 lcls = (cls_loss_pos.sum() + cls_loss_hard_neg.sum()) / (total_positives + 1e-16)
#             else:
#                 lcls = cls_loss_pos.sum() / (total_positives + 1e-16) if len(cls_loss_pos) > 0 else torch.tensor(0.0, device=self.device)

#         if self.autobalance:
#             self.balance = [x / self.balance[self.ssi] for x in self.balance]
#         lbox *= self.hyp["box"]
#         lobj *= self.hyp["obj"]
#         lcls *= self.hyp["cls"]
#         bs = tobj.shape[0]  # batch size

#         return (lbox + lobj + lcls + lcenter*0.1) * bs, torch.cat((lbox, lobj, lcls)).detach()

#     def build_targets(self, p, targets):
#         """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
#         indices, and anchors.
#         """
#         na, nt = self.na, targets.shape[0]  # number of anchors, targets
#         tcls, tbox, indices, anch = [], [], [], []
#         gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
#         ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
#         targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

#         g = 0.5  # bias
#         off = (
#             torch.tensor(
#                 [
#                     [0, 0],
#                     [1, 0],
#                     [0, 1],
#                     [-1, 0],
#                     [0, -1],  # j,k,l,m
#                     # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
#                 ],
#                 device=self.device,
#             ).float()
#             * g
#         )  # offsets

#         for i in range(self.nl):
#             anchors, shape = self.anchors[i], p[i].shape
#             gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

#             # Match targets to anchors
#             t = targets * gain  # shape(3,n,7)
#             if nt:
#                 # Matches
#                 r = t[..., 4:6] / anchors[:, None]  # wh ratio
#                 j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
#                 # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
#                 t = t[j]  # filter

#                 # Offsets
#                 gxy = t[:, 2:4]  # grid xy
#                 gxi = gain[[2, 3]] - gxy  # inverse
#                 j, k = ((gxy % 1 < g) & (gxy > 1)).T
#                 l, m = ((gxi % 1 < g) & (gxi > 1)).T
#                 j = torch.stack((torch.ones_like(j), j, k, l, m))
#                 t = t.repeat((5, 1, 1))[j]
#                 offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
#             else:
#                 t = targets[0]
#                 offsets = 0

#             # Define
#             bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
#             a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
#             gij = (gxy - offsets).long()
#             gi, gj = gij.T  # grid indices

#             # Append
#             indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
#             tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
#             anch.append(anchors[a])  # anchors
#             tcls.append(c)  # class

#         return tcls, tbox, indices, anch