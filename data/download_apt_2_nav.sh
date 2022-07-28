# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cd data
echo "Downloading the sequence and ground truth SDF ..."
gdown https://drive.google.com/drive/folders/1GTDhL0aXKv--YO0p6XLfYgSnMgzg1dXX?usp=sharing --folder
mv data_small/seqs.zip seqs.zip
mv data_small/gt_sdfs.zip gt_sdfs.zip
rm -R data_small
echo "Start unzipping ..."
unzip seqs.zip
unzip gt_sdfs.zip
mv 1seq seqs
mv 1gt_sdfs gt_sdfs
rm seqs.zip
rm gt_sdfs.zip
cd ..
echo "Sequence apt_2_nav is ready!"
