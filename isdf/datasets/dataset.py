# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import os
# import pika
from scipy.spatial.transform import Rotation as R

# import needed only when running with ROS
try:
    from isdf.ros_utils import node
except ImportError:
    print('Did not import ROS node.')

class ReplicaDataset(Dataset):
    def __init__(
        self,
        root_dir,
        traj_file=None,
        rgb_transform=None,
        depth_transform=None,
        noisy_depth=False,
        col_ext=".jpg",
        distortion_coeffs=None,
        camera_matrix=None,
    ):

        self.Ts = None
        if traj_file is not None:
            self.Ts = np.loadtxt(traj_file).reshape(-1, 4, 4)
        self.root_dir = root_dir
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.col_ext = col_ext
        self.noisy_depth = noisy_depth

    def __len__(self):
        return self.Ts.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        s = f"{idx:06}"  # int variable
        if self.noisy_depth:
            depth_file = os.path.join(self.root_dir, "ndepth" + s + ".png")
        else:
            depth_file = os.path.join(self.root_dir, "depth" + s + ".png")
        rgb_file = os.path.join(self.root_dir, "frame" + s + self.col_ext)

        depth = cv2.imread(depth_file, -1)
        image = cv2.imread(rgb_file)

        T = None
        if self.Ts is not None:
            T = self.Ts[idx]

        sample = {"image": image, "depth": depth, "T": T}

        if self.rgb_transform:
            sample["image"] = self.rgb_transform(sample["image"])

        if self.depth_transform:
            sample["depth"] = self.depth_transform(sample["depth"])

        return sample


class ScanNetDataset(Dataset):
    def __init__(
        self,
        root_dir,
        traj_file,
        rgb_transform=None,
        depth_transform=None,
        col_ext=None,
        noisy_depth=None,
        distortion_coeffs=None,
        camera_matrix=None,
    ):

        self.root_dir = root_dir
        self.rgb_dir = os.path.join(root_dir, "frames", "color/")
        self.depth_dir = os.path.join(root_dir, "frames", "depth/")
        if traj_file is not None:
            self.Ts = np.loadtxt(traj_file).reshape(-1, 4, 4)
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.col_ext = col_ext

    def __len__(self):
        return self.Ts.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        depth_file = self.depth_dir + str(idx) + ".png"
        rgb_file = self.rgb_dir + str(idx) + self.col_ext

        depth = cv2.imread(depth_file, -1)
        image = cv2.imread(rgb_file)

        T = None
        if self.Ts is not None:
            T = self.Ts[idx]

        sample = {"image": image, "depth": depth, "T": T}

        if self.rgb_transform:
            sample["image"] = self.rgb_transform(sample["image"])

        if self.depth_transform:
            sample["depth"] = self.depth_transform(sample["depth"])

        return sample


class SceneCache(Dataset):
    def __init__(
        self,
        dataset_format,
        root_dir,
        traj_file,
        keep_ixs=None,
        rgb_transform=None,
        depth_transform=None,
        noisy_depth=False,
        col_ext=".jpg",
        distortion_coeffs=None,
        camera_matrix=None,
    ):

        self.dataset_format = dataset_format
        self.Ts = np.loadtxt(traj_file).reshape(-1, 4, 4)
        self.root_dir = root_dir
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.samples = []

        if keep_ixs is not None:
            keep_ixs.sort()
        self.keep_ixs = keep_ixs

        print("Loading scene cache dataset for evaluation...")
        for idx in range(self.Ts.shape[0]):
            if keep_ixs is not None:
                if idx not in keep_ixs:
                    continue

            if dataset_format == "replicaCAD":
                s = f"{idx:06}"  # int variable
                if noisy_depth:
                    depth_file = self.root_dir + "/ndepth" + s + ".png"
                else:
                    depth_file = self.root_dir + "/depth" + s + ".png"
                rgb_file = self.root_dir + "/frame" + s + col_ext
            elif dataset_format == "ScanNet":
                depth_file = root_dir + "/frames/depth/" + str(idx) + ".png"
                rgb_file = root_dir + "/frames/color/" + str(idx) + col_ext

            depth = cv2.imread(depth_file, -1)
            image = cv2.imread(rgb_file)

            if self.rgb_transform:
                image = self.rgb_transform(image)

            if self.depth_transform:
                depth = self.depth_transform(depth)

            self.samples.append((image, depth, self.Ts[idx]))

        self.samples = np.array(self.samples)
        print("Len cached dataset", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def get_all(self):
        idx = np.arange(self.__len__())

        image = np.concatenate(([x[None, :] for x in self.samples[idx, 0]]))
        depth = np.concatenate(([x[None, :] for x in self.samples[idx, 1]]))
        T = np.concatenate(([x[None, :] for x in self.samples[idx, 2]]))

        sample = {
            "image": image,
            "depth": depth,
            "T": T
        }

        return sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.keep_ixs is not None:
            idx = [x for x in idx if x in self.keep_ixs]
            idx = np.array([np.where(self.keep_ixs == x)[0][0] for x in idx])

        image = np.concatenate(([x[None, :] for x in self.samples[idx, 0]]))
        depth = np.concatenate(([x[None, :] for x in self.samples[idx, 1]]))
        T = np.concatenate(([x[None, :] for x in self.samples[idx, 2]]))

        sample = {
            "image": image,
            "depth": depth,
            "T": T
        }

        return sample


# Consume RGBD + pose data from ROS node
class ROSSubscriber(Dataset):
    def __init__(
        self,
        dataset_format=None,
        root_dir=None,
        traj_file=None,
        keep_ixs=None,
        rgb_transform=None,
        depth_transform=None,
        noisy_depth=False,
        col_ext=None,
        distortion_coeffs=None,
        camera_matrix=None,
    ):
        crop = False
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

        self.distortion_coeffs = np.array(distortion_coeffs)
        self.camera_matrix = camera_matrix

        torch.multiprocessing.set_start_method('spawn', force=True)
        self.queue = torch.multiprocessing.Queue(maxsize=1)

        process = torch.multiprocessing.Process(
            target=node.iSDFNode,
            args=(self.queue, crop),
        )
        process.start()

    def __len__(self):
        return 1000000000

    def __getitem__(self, idx):
        data = None
        while data is None:
            data = node.get_latest_frame(self.queue)

            if data is not None:
                image, depth, Twc = data

                if self.rgb_transform:
                    image = self.rgb_transform(image)
                if self.depth_transform:
                    depth = self.depth_transform(depth)

                    # undistort depth, using nn rather than linear interpolation
                    img_size = (depth.shape[1], depth.shape[0])
                    map1, map2 = cv2.initUndistortRectifyMap(
                        self.camera_matrix, self.distortion_coeffs, np.eye(3),
                        self.camera_matrix, img_size, cv2.CV_32FC1)
                    depth = cv2.remap(depth, map1, map2, cv2.INTER_NEAREST)

                sample = {
                    "image": image,
                    "depth": depth,
                    "T": Twc,
                }
                return sample


# Consume RGBD + pose data from an IOS device with a Lidar
# class ARKit(Dataset):
#     def __init__(
#         self,
#         dataset_format=None,
#         root_dir=None,
#         traj_file=None,
#         keep_ixs=None,
#         rgb_transform=None,
#         depth_transform=None,
#         noisy_depth=False,
#         col_ext=None,
#         distortion_coeffs=None,
#         camera_matrix=None,
#     ):
#         self.rgb_transform = rgb_transform
#         self.depth_transform = depth_transform

#         # Connect to MQT Broker
#         user_and_pass = os.getenv('rabbitMQUserNameAndPassword', 'guest')
#         credentials = pika.PlainCredentials(user_and_pass, user_and_pass)
#         self.connection = pika.BlockingConnection(
#             pika.ConnectionParameters(
#                 host=os.getenv('rabbitMQBroker', 'localhost'),
#                 credentials=credentials,
#             )
#         )

#         self.rgb_channel = self.connection.channel()
#         self.rgb_channel.queue_declare(
#             queue='rgb_frame', arguments={"x-max-length": 3})

#         self.depth_channel = self.connection.channel()
#         self.depth_channel.queue_declare(
#             queue='depth_frame', arguments={"x-max-length": 3})

#     def __len__(self):
#         return 1000000000

#     def __getitem__(self, idx):
#         # get RGB frame
#         _, _, rgb_bytes = next(
#             self.rgb_channel.consume(queue='rgb_frame', auto_ack=True))
#         self.rgb_channel._queue_consumer_generator.pending_events.clear()
#         self.rgb_channel.queue_purge('rgb_frame')

#         img_array = np.asarray(bytearray(rgb_bytes), dtype=np.uint8)
#         image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

#         # Get Pose + Depth frame
#         _, _, depth_bytes = next(
#             self.depth_channel.consume(
#                 queue='depth_frame',
#                 auto_ack=True,
#                 arguments={"x-max-length": 3},
#             )
#         )
#         self.depth_channel._queue_consumer_generator.pending_events.clear()
#         self.depth_channel.queue_purge('depth_frame')

#         # Pose and depth buffer, the first 16x 32 bytes are the
#         # transformation matrix, next 4x 32 bytes are the camera intrinsics,
#         # the rest is the 256x192 depth frame
#         raw_msg = np.frombuffer(depth_bytes, dtype=np.float32)
#         arkit_pose = raw_msg[:16].reshape((4, 4)).transpose()
#         intrinsics = raw_msg[16: 20]  # fx, fy, cx, cy
#         depth = raw_msg[20:].reshape((192, 256))

#         # Transform the arkit pose to image coordinate frame
#         # Rotate around x-axis by 180deg
#         tf = np.array([[1.0, 0.0, 0.0],
#                        [0.0, -1.0, 0.0],
#                        [0.0, 0.0, -1.0]])
#         tf = np.vstack((tf, [0.0, 0.0, 0.0]))
#         tf = np.hstack((tf, [[0.0], [0.0], [0.0], [1.0]]))
#         Twc = np.matmul(tf, arkit_pose)

#         # Offsetting the rotation around the x-axis to get all
#         # the rotaions starting in the same range
#         rot = R.from_matrix(Twc[0:3, 0:3]).as_euler('xyz', degrees=True)
#         rot[0] += 180
#         Twc[0:3, 0:3] = R.from_euler('xyz', rot, degrees=True).as_matrix()

#         if self.rgb_transform:
#             image = self.rgb_transform(image)
#         if self.depth_transform:
#             depth = self.depth_transform(depth)

#         sample = {
#             "image": image,
#             "depth": depth,
#             "T": Twc,
#             "intrinsics": intrinsics
#         }

#         return sample
