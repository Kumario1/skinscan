# Lesion-crop type classifier (Stage 2, spec §4.2).
# Pure parts (CLASSES, crop_with_context, StubClassifier) run with numpy+PIL only;
# torch is imported lazily so the pipeline and its tests need no weights and no GPU.
import numpy as np
from PIL import Image

# Alphabetical on purpose: torchvision ImageFolder sorts class dirs alphabetically,
# so this order IS the training label order. Do not reorder.
CLASSES = ["comedonal", "cystic", "inflammatory", "not_acne", "post_acne_mark"]


def crop_with_context(image, box, pad=1.5, size=112):
    """Square crop around a detector box with context padding (spec §4.2).

    image: HxWx3 uint8 array. box: (x, y, w, h[, conf]) in pixel coords.
    Pads the box by `pad`, squares it, replicate-pads at image edges
    (never shifts the box), resizes to size x size.
    """
    x, y, w, h = box[:4]
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(max(w, h) * pad, 2.0)  # guard degenerate boxes
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + int(round(side)), y0 + int(round(side))
    H, W = image.shape[:2]
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - W), max(0, y1 - H)
    if pl or pt or pr or pb:
        image = np.pad(image, ((pt, pb), (pl, pr), (0, 0)), mode="edge")
        x0, y0, x1, y1 = x0 + pl, y0 + pt, x1 + pl, y1 + pt
    crop = image[y0:y1, x0:x1]
    return np.asarray(Image.fromarray(crop).resize((size, size), Image.BILINEAR))


def build_net(pretrained=False):
    """MobileNetV3-small with a 5-class head. Shared by training notebook and
    inference so the architecture can't drift between the two."""
    import torch.nn as nn
    import torchvision
    net = torchvision.models.mobilenet_v3_small(
        weights="IMAGENET1K_V1" if pretrained else None
    )
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, len(CLASSES))
    return net


class LesionClassifier:
    """Real model. Weights required — missing weights is a hard error, never a
    silent mislabel (spec §7)."""

    # must match the notebook's training transform
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, weights_path, device="cpu"):
        import torch
        self._torch = torch
        net = build_net(pretrained=False)
        net.load_state_dict(torch.load(weights_path, map_location=device))
        net.eval()
        self.net = net.to(device)
        self.device = device

    def predict(self, crop):
        """crop: size x size x 3 uint8 -> {class: prob}."""
        t = self._torch
        x = (crop.astype(np.float32) / 255.0 - self._MEAN) / self._STD
        x = t.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with t.no_grad():
            probs = t.softmax(self.net(x), dim=1)[0].cpu().numpy()
        return dict(zip(CLASSES, probs.tolist()))


class StubClassifier:
    """Same interface, fixed probs — lets assemble.py and its tests run with no
    weights and no torch (spec §4.2, D-007)."""

    def __init__(self, probs=None):
        p = np.asarray(
            probs if probs is not None else [0.7, 0.05, 0.1, 0.1, 0.05],
            dtype=np.float32,
        )
        self.probs = p / p.sum()

    def predict(self, crop):
        return dict(zip(CLASSES, self.probs.tolist()))


if __name__ == "__main__":
    # ponytail: assert-based self-check, no framework
    img = np.zeros((100, 200, 3), np.uint8)
    img[40:60, 90:110] = 255  # bright "lesion" so we can assert the crop caught it
    c = crop_with_context(img, (90, 40, 20, 20))
    assert c.shape == (112, 112, 3) and c.dtype == np.uint8
    assert c.mean() > 80, "crop missed the lesion"
    corner = crop_with_context(img, (0, 0, 10, 10, 0.9))  # edge box + conf tail
    assert corner.shape == (112, 112, 3)
    tiny = crop_with_context(img, (50, 50, 0, 0))  # degenerate box must not crash
    assert tiny.shape == (112, 112, 3)
    p = StubClassifier().predict(c)
    assert set(p) == set(CLASSES) and abs(sum(p.values()) - 1.0) < 1e-5
    assert max(p, key=p.get) == "comedonal"
    print("ok")
