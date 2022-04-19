import os
import queue
import numpy as np
from scipy.spatial.transform import Rotation
import rospy
import trimesh
# import cv2
# import imgviz
# from time import perf_counter

from orb_slam3_ros_wrapper.msg import frame


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
