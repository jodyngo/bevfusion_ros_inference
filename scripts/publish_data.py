#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, Image, PointField
import sensor_msgs.point_cloud2 as pc2
from PIL import Image as PILImage
import torch
import os
import glob
import cv2

class DataPublisher:
    def __init__(self, point_cloud_dir, image_dir, publish_rate=10):
        rospy.init_node('data_publisher', anonymous=True)
        
        # Publishers
        self.pc_pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=10)
        self.img_pub = rospy.Publisher('/image', Image, queue_size=10)
        
        
        # Data directories
        self.point_cloud_dir = point_cloud_dir
        self.image_dir = image_dir
        
        # Load file lists
        self.pc_files = sorted(glob.glob(os.path.join(point_cloud_dir, '*.bin')))
        self.img_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        
        if not self.pc_files or not self.img_files:
            rospy.logerr("No point cloud or image files found in the specified directories")
            raise RuntimeError("No data files found")
        
        # Ensure the number of files match
        if len(self.pc_files) != len(self.img_files):
            rospy.logwarn(f"Mismatch: {len(self.pc_files)} point clouds, {len(self.img_files)} images")
        
        self.rate = rospy.Rate(publish_rate)  # Publish rate in Hz
        rospy.loginfo("Data publisher initialized")

    def load_point_cloud(self, file_path: str) -> np.ndarray:
        """Load point cloud from binary file."""
        dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32), ('ObjTag', np.uint32)]
        points = np.fromfile(file_path, dtype=dtype)
        x = points['x']
        y = points['y']
        z = points['z']
        intensity = points['intensity']
        obj_tag = points['ObjTag']
        points_array = np.stack((x, y, z, intensity, obj_tag), axis=-1)
        points_array = points_array.reshape(-1, 5)
        return points_array

    def load_image(self, file_path: str, color_type='color', to_float32=True) -> np.ndarray:
        """Load image from file."""
        with open(file_path, 'rb') as f:
            img = PILImage.open(f)
            if color_type == 'color':
                img = img.convert('RGB')
            elif color_type == 'grayscale':
                img = img.convert('L')
            else:
                raise ValueError(f"Unsupported color_type: {color_type}")

            img_np = np.array(img)

            if color_type == 'grayscale':
                img_np = np.expand_dims(img_np, axis=-1)  # Shape: (H, W, 1)
            else:
                # Ensure channel-last format (H, W, C) for RGB
                if img_np.shape[0] in [1, 3]:  # Handle (C, H, W) case
                    img_np = np.transpose(img_np, (1, 2, 0))

            if to_float32:
                img_np = img_np.astype(np.float32)

            img_tensor = torch.from_numpy(img_np)  # Keep [H, W, C]
            img_tensor = img_tensor.unsqueeze(0).permute(0, 3, 1, 2)  # Shape: [1, C, H, W]
            img_np = img_tensor.cpu().numpy()
            return img_np

    def create_point_cloud_msg(self, points: np.ndarray, timestamp, frame_id="base_link") -> PointCloud2:
        """Convert numpy point cloud to sensor_msgs/PointCloud2 with x, y, z, intensity, ObjTag."""
        header = rospy.Header()
        header.stamp = timestamp
        header.frame_id = frame_id
        
        # Define fields: x, y, z, intensity (float32), ObjTag (uint32)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='ObjTag', offset=16, datatype=PointField.UINT32, count=1),
        ]
        
        # Ensure points have correct shape (N, 5)
        if points.shape[1] != 5:
            rospy.logerr(f"Invalid point cloud shape: {points.shape}, expected (N, 5)")
            return None
            
        # Convert points to list of tuples for pc2.create_cloud
        point_list = [(p[0], p[1], p[2], p[3], int(p[4])) for p in points]
        
        pc_msg = pc2.create_cloud(header, fields, point_list)
        rospy.loginfo(f"Created PointCloud2 with {len(point_list)} points, fields: x, y, z, intensity, ObjTag")
        return pc_msg

    def cv2_to_imgmsg(self, cv_image):
        img_msg = Image()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = 0
        img_msg.data = cv_image.tobytes()
        img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
        return img_msg

    def create_image_msg(self, file_path: str, timestamp, frame_id="base_link") -> Image:
        """Load image and convert to sensor_msgs/Image with bgr8 encoding."""
        # Load image as RGB and convert to BGR for ROS
        img_np = self.load_image(file_path, color_type='color', to_float32=False)  # Load as uint8
        img_np = img_np[0].transpose(1, 2, 0)  # Convert from [1, C, H, W] to [H, W, C]
        img_np = img_np.astype(np.uint8)  # Ensure uint8 for bgr8
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        
        img_msg = self.cv2_to_imgmsg(img_bgr)
        img_msg.header.stamp = timestamp
        img_msg.header.frame_id = frame_id
        return img_msg

    def publish_data(self):
        while not rospy.is_shutdown():
            for pc_file, img_file in zip(self.pc_files, self.img_files):
                try:
                    if rospy.is_shutdown():
                        break
                    points = self.load_point_cloud(pc_file)
                    if points.shape[1] != 5:
                        rospy.logwarn(f"Invalid point cloud format in {pc_file}: {points.shape}")
                        continue
                    timestamp = rospy.Time.now()
                    pc_msg = self.create_point_cloud_msg(points, timestamp)
                    img_msg = self.create_image_msg(img_file, timestamp)
                    self.pc_pub.publish(pc_msg)
                    self.img_pub.publish(img_msg)
                    rospy.loginfo(f"Published: {os.path.basename(pc_file)}, {os.path.basename(img_file)}")
                    rospy.loginfo(f"Point cloud shape: {points.shape}, Image shape: {img_msg.height}x{img_msg.width}x{3}")
                    self.rate.sleep()
                except Exception as e:
                    rospy.logerr(f"Error publishing {pc_file} or {img_file}: {e}")
                    continue
            rospy.loginfo("Reached end of file list, restarting loop")

def main():
    try:
        point_cloud_dir = '/var/local/home/thungo/bevfusion_ws/src/bevfusion_onnx/scripts/data/0149/val/lidar_all'
        image_dir ='/var/local/home/thungo/bevfusion_ws/src/bevfusion_onnx/scripts/data/0149/val/image_all'
        # point_cloud_dir = '/home/nvidia/bevfusion_ws/src/bevfusion_onnx/scripts/data/0149/val/lidar_all'
        # image_dir ='/home/nvidia/bevfusion_ws/src/bevfusion_onnx/scripts/data/0149/val/image_all'
        rate = 10.0
        publisher = DataPublisher(point_cloud_dir, image_dir, rate)
        publisher.publish_data()
    except rospy.ROSInterruptException:
        rospy.loginfo("Data publisher terminated")

if __name__ == '__main__':
    main()
