# serve.py - LitServe inference API for the SA-RPN acne detector.
# Run inside sa-rpn-env, from the repo dir (CPU is fine for testing, GPU for load):
#   cd /teamspace/studios/this_studio/acnedetection
#   PYTHONPATH="$PWD" \
#     CKPT=/teamspace/studios/this_studio/skinscan_out/work_dir/epoch_15.pth \
#     /teamspace/studios/this_studio/sa-rpn-env/bin/python \
#     /teamspace/studios/this_studio/serve.py
import base64
import io
import os

import numpy as np
from PIL import Image
import litserve as ls  # pin litserve==0.1.5 (last release that runs on Python 3.8)
from mmdet.apis import init_detector, inference_detector

CONFIG = os.environ.get("CFG", "configs/skin/skin_config.py")
CKPT = os.environ.get("CKPT", "skinscan_out/work_dir/epoch_15.pth")
SCORE_THR = float(os.environ.get("SCORE_THR", "0.3"))


def _dedupe(dets, thr=0.5):
    # Class-agnostic suppression on the final detections. Uses intersection over
    # the SMALLER box (not IoU) so a small box nested inside a big one is culled
    # too - the exact duplicate mode seen in validation. dets must be sorted by
    # score desc. ponytail: O(N^2) greedy, N is ~tens per image at most.
    keep = []
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        suppressed = False
        for k in keep:
            kx1, ky1, kx2, ky2 = k["bbox"]
            iw = min(x2, kx2) - max(x1, kx1)
            ih = min(y2, ky2) - max(y1, ky1)
            if iw <= 0 or ih <= 0:
                continue
            karea = (kx2 - kx1) * (ky2 - ky1)
            smaller = min(area, karea) or 1.0
            if (iw * ih) / smaller > thr:
                suppressed = True
                break
        if not suppressed:
            keep.append(d)
    return keep


class AcneDetector(ls.LitAPI):
    def setup(self, device):
        self.model = init_detector(CONFIG, CKPT, device=device)
        self.classes = self.model.CLASSES  # class names come from the checkpoint meta

    def decode_request(self, request):
        img = Image.open(io.BytesIO(base64.b64decode(request["image"]))).convert("RGB")
        return np.asarray(img)[:, :, ::-1]  # RGB -> BGR for mmdet

    # ponytail: images differ in size, so keep the batch a plain list (no np.stack).
    # inference_detector handles both a single ndarray and a list, so predict works
    # whether or not batching is active.
    def batch(self, items):
        return list(items)

    def predict(self, imgs):
        return inference_detector(self.model, imgs)

    def unbatch(self, results):
        return list(results)

    def encode_response(self, result):
        bbox_result = result[0] if isinstance(result, tuple) else result
        dets = []
        for cls_id, boxes in enumerate(bbox_result):
            for b in boxes:
                score = float(b[4])
                if score < SCORE_THR:
                    continue
                dets.append({
                    "label": self.classes[cls_id],
                    "score": round(score, 4),
                    "bbox": [round(float(v), 1) for v in b[:4]],  # x1,y1,x2,y2
                })
        dets.sort(key=lambda d: d["score"], reverse=True)
        dets = _dedupe(dets)
        return {"count": len(dets), "detections": dets}


if __name__ == "__main__":
    import torch

    # In litserve 0.1.5 the batching args live on LitServer, not LitAPI.
    # accelerator="auto" probes torch.backends.mps, which torch 1.9 lacks -
    # pick cpu/cuda ourselves instead.
    accel = "cuda" if torch.cuda.is_available() else "cpu"
    api = AcneDetector()
    ls.LitServer(
        api, accelerator=accel, devices=1, max_batch_size=4, batch_timeout=0.05
    ).run(port=8000)
