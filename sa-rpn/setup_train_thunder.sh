#!/usr/bin/env bash
# AcneSCU SA-RPN training on a Thunder Compute A6000 (Ampere, sm_86).
# Mirrors the proven Colab notebook cells, minus Colab-isms.
#
# BEFORE RUNNING, put these two files in the current directory:
#   - AcneSCU.v1*.zip
#   - prepare_acnescu_sa_rpn.py
# Then:  chmod +x setup_train_thunder.sh && ./setup_train_thunder.sh
# Run it inside tmux so training survives SSH disconnects:  tmux new -s train
set -euxo pipefail

HOME_DIR="$HOME"
ENV="$HOME_DIR/sa-rpn-env"
REPO="$HOME_DIR/acnedetection"
RAW="$HOME_DIR/acnescu_v1"
OUT="$HOME_DIR/skinscan_out"        # scp this back before deleting the instance
PREPARED="$OUT/coco"
WORK_DIR="$OUT/work_dir"
CKPT="$HOME_DIR/mask_rcnn_r50_fpn_1x_coco.pth"
mkdir -p "$OUT" "$WORK_DIR"

# --- locate the two inputs you scp'd up -------------------------------------
ZIP=$(ls "$HOME_DIR"/AcneSCU.v1*.zip 2>/dev/null | head -n1 || true)
CONV=$(ls "$HOME_DIR"/prepare_acnescu_sa_rpn.py 2>/dev/null | head -n1 || true)
test -n "$ZIP"  || { echo "ERROR: AcneSCU.v1*.zip not found in $HOME_DIR"; exit 1; }
test -n "$CONV" || { echo "ERROR: prepare_acnescu_sa_rpn.py not found in $HOME_DIR"; exit 1; }

# --- 1. legacy env: python 3.8 / torch 1.9 cu111 / mmcv 1.4 -----------------
SUDO=$(command -v sudo || true)   # ponytail: RunPod/Vast containers run as root without sudo
if [ ! -x /usr/bin/unzip ] && [ ! -x /bin/unzip ]; then $SUDO apt-get update -y && $SUDO apt-get install -y unzip; fi
if [ ! -x "$HOME_DIR/bin/micromamba" ]; then
  mkdir -p "$HOME_DIR/bin"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$HOME_DIR" bin/micromamba
fi
if [ ! -x "$ENV/bin/python" ]; then
  "$HOME_DIR/bin/micromamba" create -y -p "$ENV" python=3.8 pip
  "$ENV/bin/pip" install 'numpy==1.23.5' 'torch==1.9.0+cu111' 'torchvision==0.10.0+cu111' \
      -f https://download.pytorch.org/whl/torch_stable.html
  "$ENV/bin/pip" install 'mmcv-full==1.4.0' \
      -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
  "$ENV/bin/pip" install pycocotools terminaltables scipy pillow matplotlib \
      opencv-python-headless 'yapf==0.31.0' addict pyyaml
fi
PY="$ENV/bin/python"
"$PY" - <<'PYEOF'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())
print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
assert torch.cuda.is_available(), 'no CUDA device visible'
PYEOF

# --- 2. author code + exact masked crops ------------------------------------
if [ ! -d "$REPO" ]; then
  git clone --depth 1 https://github.com/pingguokiller/acnedetection.git "$REPO"
fi
if [ ! -d "$RAW" ] || [ -z "$(ls -A "$RAW" 2>/dev/null)" ]; then
  mkdir -p "$RAW"; unzip -q -o "$ZIP" -d "$RAW"
fi
if [ ! -f "$PREPARED/dataset_metadata.json" ]; then
  "$PY" "$CONV" --voc-root "$RAW" --out "$PREPARED" --test-count 28 --seed 42
fi
cat "$PREPARED/dataset_metadata.json"

# --- 3. patch machine paths / eval target (idempotent), fetch COCO init ------
if [ ! -f "$CKPT" ]; then
  wget -q -O "$CKPT" https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth
fi
test "$(stat -c%s "$CKPT")" -gt 1000000 || { echo "COCO checkpoint truncated"; exit 1; }

CFG_DIR="$REPO/configs/skin" PREPARED="$PREPARED" CKPT="$CKPT" "$PY" - <<'PYEOF'
import os
from pathlib import Path
cfg_dir = Path(os.environ['CFG_DIR']); prepared = os.environ['PREPARED']; ckpt = os.environ['CKPT']
def patch(name, repls):
    p = cfg_dir / name; t = p.read_text()
    for old, new in repls:
        if new in t: continue
        assert old in t, f'{name}: expected text not found:\n  {old}'
        t = t.replace(old, new)
    p.write_text(t)
patch('coco_instance.py', [
    ("data_root = '/home/zhangjw/research/skin_data/coco/'", f"data_root = '{prepared}/'"),
    ("ann_file=data_root + 'annotations/all_whole.json'",    "ann_file=data_root + 'annotations/val_crop_1024.json'"),
    ("img_prefix=data_root + 'images/all_whole/'",           "img_prefix=data_root + 'images/val_crop_1024/'"),
])
patch('default_runtime.py', [('load_from = None', f"load_from = '{ckpt}'")])
# A6000 has 48 GB: keep the paper's batch 2 (do NOT drop to 1)
sched = (cfg_dir / 'schedule_2x.py').read_text()
for e in ['lr=0.002','momentum=0.9','weight_decay=0.0001','max_epochs=15']:
    assert e in sched, f'schedule_2x.py missing {e}'
print('config patched OK ->', prepared)
PYEOF

# --- 4. train (PYTHONPATH makes tools/train.py find the in-repo mmdet) -------
CMD=("$PY" tools/train.py configs/skin/skin_config.py --work_dir "$WORK_DIR" --seed 42 --deterministic)
if [ -f "$WORK_DIR/latest.pth" ]; then
  echo "Resuming from $WORK_DIR/latest.pth"
  CMD+=(--resume-from "$WORK_DIR/latest.pth")
fi
cd "$REPO"
PYTHONPATH="$REPO" "${CMD[@]}"

echo "DONE. Checkpoints in $WORK_DIR"
echo "From your laptop, pull results back with:"
echo "  tnr scp 0:$WORK_DIR/epoch_15.pth ./"
