# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cd data
echo "Downloading the realsense_franka data ..."
gdown https://drive.google.com/drive/folders/1tBk0W4wmytbISqSFg7A7eDvGcBt6uouJ?usp=sharing --folder
echo "Start unzipping ..."
unzip -q franka_iSDF/realsense_franka.zip
rm -R franka_iSDF
cd ..
echo "Sequence realsense_franka is ready!"
