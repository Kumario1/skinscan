#!/usr/bin/env bash
# One-time publish: private dataset (370MB zip + converter) then the training kernel.
# Prereq: ~/.kaggle/kaggle.json (kaggle.com -> Settings -> API -> Create New Token)
set -euo pipefail
cd "$(dirname "$0")"

USER=$(kaggle config view | sed -n 's/^- username: //p')
sed -i '' "s/KAGGLE_USERNAME/$USER/g" dataset/dataset-metadata.json kernel/kernel-metadata.json

# create on first run, version on reruns
kaggle datasets create -p dataset || kaggle datasets version -p dataset -m "update" --delete-old-versions

echo "Waiting 90s for Kaggle to process the dataset before pushing the kernel..."
sleep 90
kaggle kernels push -p kernel

echo "Training started (runs in background on Kaggle, no browser needed):"
echo "  https://www.kaggle.com/code/$USER/acnescu-sa-rpn-train"
echo "Check status:  kaggle kernels status $USER/acnescu-sa-rpn-train"
echo "Grab results:  kaggle kernels output $USER/acnescu-sa-rpn-train -p ./kaggle_out"
