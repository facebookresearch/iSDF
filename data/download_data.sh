# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cd data
echo "Downloading the sequences and ground truth SDFs ..."
gdown https://drive.google.com/drive/folders/1nzAVDInjDwt_GFehyhkOZvXrRJ33FCaR?usp=sharing --folder
echo "Start unzipping ..."
unzip seqs.zip
unzip gt_sdfs.zip
unzip eval_pts.zip
rm seqs.zip
rm gt_sdfs.zip
rm eval_pts.zip
cd ..
echo "Dataset is ready!"
