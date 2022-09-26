# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import queue
import numpy as np
from scipy.spatial.transform import Rotation
import rospy
import trimesh
import cv2
# import imgviz
# from time import perf_counter

from orb_slam3_ros_wrapper.msg import frame
from sensor_msgs.msg import Image # ROS message type
from geometry_msgs.msg import Pose # ROS message type
from matplotlib import pyplot as plt

class iSDFNode:
    
    def __init__(self, queue, crop=False) -> None:
        print("iSDF Node: starting", os.getpid())
        print("Waiting for first frame...")

        self.queue = queue

        self.crop = crop

        # self.first_pose_inv = None
        # self.world_transform = trimesh.transformations.rotation_matrix(
        #         np.deg2rad(-90), [1, 0, 0]) @ trimesh.transformations.rotation_matrix(
        #         np.deg2rad(90), [0, 1, 0])

        rospy.init_node("isdf", anonymous=True)
        rospy.Subscriber("/frames", frame, self.callback)
        rospy.spin()

    def callback(self, msg):
        if self.queue.full():
            return

        # start = perf_counter()

        rgb_np = np.frombuffer(msg.rgb.data, dtype=np.uint8)
        rgb_np = rgb_np.reshape(msg.rgb.height, msg.rgb.width, 3)
        rgb_np = rgb_np[..., ::-1]

        depth_np = np.frombuffer(msg.depth.data, dtype=np.uint16)
        depth_np = depth_np.reshape(msg.depth.height, msg.depth.width)

        # Crop images to remove the black edges after calibration
        if self.crop:
            w = msg.rgb.width
            h = msg.rgb.height
            mw = 40
            mh = 20
            rgb_np = rgb_np[mh:(h - mh), mw:(w - mw)]
            depth_np = depth_np[mh:(h - mh), mw:(w - mw)]

        # depth_viz = imgviz.depth2rgb(
        #     depth_np.astype(np.float32) / 1000.0)[..., ::-1]
        # viz = np.hstack((rgb_np, depth_viz))
        # cv2.imshow('rgbd', viz)
        # cv2.waitKey(1)

        # Formatting camera pose as a transformation matrix w.r.t world frame
        position = msg.pose.position
        quat = msg.pose.orientation
        trans = np.asarray([[position.x], [position.y], [position.z]])
        rot = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
        camera_transform = np.concatenate((rot, trans), axis=1)
        camera_transform = np.vstack((camera_transform, [0.0, 0.0, 0.0, 1.0]))

        camera_transform = np.linalg.inv(camera_transform)

        # if self.first_pose_inv is None: 
        #     self.first_pose_inv = np.linalg.inv(camera_transform)
        # camera_transform = self.first_pose_inv @ camera_transform

        # camera_transform = camera_transform @ self.world_transform

        try:
            self.queue.put(
                (rgb_np.copy(), depth_np.copy(), camera_transform.copy()),
                block=False,
            )
        except queue.Full:
            pass

        del rgb_np
        del depth_np
        del camera_transform

        # ed = perf_counter()
        # print("sub time: ", ed - start)

class iSDFFrankaNode:
    def __init__(self, queue, crop=False, ext_calib = None) -> None:
        print("iSDF Franka Node: starting", os.getpid())
        print("Waiting for first frame...")

        self.queue = queue
        self.crop = crop
        self.camera_transform = None 

        self.cal = ext_calib

        self.rgb, self.depth, self.pose = None, None, None

        self.first_pose_inv = None

        rospy.init_node("isdf_franka")
        rospy.Subscriber("/franka/rgb", Image, self.main_callback, queue_size=1)
        rospy.Subscriber("/franka/depth", Image, self.depth_callback, queue_size=1)
        rospy.Subscriber("/franka/pose", Pose, self.pose_callback, queue_size=1)
        rospy.spin()

    def main_callback(self, msg):
        # main callback is RGB, and uses the latest depth + pose 
        # TODO: subscribe to single msg type that contains (image, depth, pose)
        rgb_np = np.frombuffer(msg.data, dtype=np.uint8)
        rgb_np = rgb_np.reshape(msg.height, msg.width, 3)
        rgb_np = rgb_np[..., ::-1]
        self.rgb = cv2.resize(rgb_np, (1280, 720), interpolation=cv2.INTER_AREA)
        
        del rgb_np

        if self.depth is None or self.pose is None: 
            return
        # self.show_rgbd(self.rgb, self.depth, 0)

        try:
            self.queue.put(
                (self.rgb.copy(), self.depth.copy(), self.pose.copy()),
                block=False,
            )
        except queue.Full:
            pass
        
    def depth_callback(self, msg):
        depth_np = np.frombuffer(msg.data, dtype=np.uint16)
        depth_np = depth_np.reshape(msg.height, msg.width)
        self.depth = cv2.resize(depth_np, (1280, 720), interpolation=cv2.INTER_AREA)
        del depth_np

    def pose_callback(self, msg):
        position = msg.position
        quat = msg.orientation
        trans = np.asarray([position.x, position.y, position.z])
        rot = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
        trans, rot = self.ee_to_cam(trans, rot)
        camera_transform = np.concatenate((rot, trans.reshape((3, 1))), axis=1)
        camera_transform = np.vstack((camera_transform, [0.0, 0.0, 0.0, 1.0]))
        self.pose = camera_transform

        del camera_transform

    def ee_to_cam(self, trans, rot):
        # transform the inverse kinematics EE pose to the realsense pose
        cam_ee_pos = np.array(self.cal[0]['camera_ee_pos'])
        cam_ee_rot = np.array(self.cal[0]['camera_ee_ori_rotvec'])
        cam_ee_rot = Rotation.from_rotvec(cam_ee_rot).as_matrix()

        camera_world_pos = trans + rot @ cam_ee_pos
        camera_world_rot = rot @ cam_ee_rot
        return camera_world_pos, camera_world_rot

def show_rgbd(rgb, depth, timestamp):
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.imshow(rgb)
    plt.title('RGB ' + str(timestamp))
    plt.subplot(2, 1, 2)
    plt.imshow(depth)
    plt.title('Depth ' + str(timestamp))
    plt.draw()
    plt.pause(1e-6)


def get_latest_frame(q):
    # Empties the queue to get the latest frame
    message = None
    while True:
        try:
            message_latest = q.get(block=False)
            if message is not None:
                del message
            message = message_latest

        except queue.Empty:
            break

    return message
