#!/usr/bin/env python

import rospy
import numpy as np
import onnx
import onnxruntime as ort
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs.point_cloud2 as pc2
import torch
from torch import nn
from typing import Tuple, List
from PIL import Image as PILImage
from bevfusion_onnx.msg import CustomDetection3DArray, CustomDetection3D
import sys
import time
from std_msgs.msg import Header, Float32

# source /var/local/home/thungo/bevfusion_ws/devel/setup.bash
# mmdetection3d/mmdet3d/models/task_modules/voxel//voxel_generator.py

class VoxelGenerator:
    """Voxel generator in numpy implementation."""
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels=120000):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size

    def generate(self, points: np.ndarray) -> Tuple[np.ndarray]:
        """Generate voxels given points."""
        return points_to_voxel(points, self._voxel_size, self._point_cloud_range,
                             self._max_num_points, True, self._max_voxels)
    
def points_to_voxel(points, voxel_size, coors_range, max_points, reverse_index, max_voxels):
    """Convert points to voxels."""
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    if reverse_index:
        voxelmap_shape = voxelmap_shape[::-1]
    
    num_points_per_voxel = np.zeros((max_voxels,), dtype=np.int32)
    coor_to_voxelidx = -np.ones(voxelmap_shape, dtype=np.int32)
    voxels = np.zeros((max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    coors = np.zeros((max_voxels, 3), dtype=np.int32)
    
    voxel_num = _points_to_voxel_reverse_kernel(points, voxel_size, coors_range,
                                            num_points_per_voxel, coor_to_voxelidx,
                                            voxels, coors, max_points, max_voxels)
    
    return voxels[:voxel_num], coors[:voxel_num], num_points_per_voxel[:voxel_num]

def _points_to_voxel_reverse_kernel(points, voxel_size, coors_range, num_points_per_voxel,
                                  coor_to_voxelidx, voxels, coors, max_points, max_voxels):
    N = points.shape[0]
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros((3,), dtype=np.int32)
    voxel_num = 0
    
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num

class ImageAug3D(nn.Module):
    def __init__(self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip,
                 is_train):
        super(ImageAug3D, self).__init__()
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, ori_shape):
        H, W = ori_shape
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(self, img, rotation, translation, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = PILImage.fromarray(img.astype('uint8'), mode='RGB')
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ])
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def forward(self, imgs) :
        ori_shape = (2168, 3848)
        new_imgs = []
        transforms = []

        resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
            ori_shape)
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)
        new_img, rotation, translation = self.img_transform(
            imgs,
            post_rot,
            post_tran,
            resize=resize,
            resize_dims=resize_dims,
            crop=crop,
            flip=flip,
            rotate=rotate,
        )
        transform = torch.eye(4)
        transform[:2, :2] = rotation
        transform[:2, 3] = translation
        new_imgs = np.array(new_img).astype(np.float32)
        # new_imgs.append(np.array(new_img).astype(np.float32))
        transforms.append(transform.numpy())
        # update the calibration matrices
        return new_imgs
    
class BEVFusionONNXNode:
    def __init__(self):
        rospy.init_node('bevfusion_onnx_node', anonymous=True)
        
        # Parameters
        self.onnx_model_path = rospy.get_param('onnx_model_path', '/var/local/home/thungo/bevfusion_ws/src/bevfusion_onnx/scripts/checkpoint/bevfusion_lidar_cam_sim.onnx')
        # self.onnx_model_path = rospy.get_param('onnx_model_path', '/home/nvidia/bevfusion_ws/src/bevfusion_onnx/scripts/checkpoint/bevfusion_lidar_cam.onnx')
        self.voxelize_cfg = {
            'max_num_points': 10,
            'point_cloud_range': [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            'voxel_size': [0.075, 0.075, 0.2],
            'max_voxels': [120000, 160000],
            'voxelize_reduce': True
        }
        self.score_threshold = rospy.get_param('~score_threshold', 0.00001)
        
        self.session = self.init_onnx_model()
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        
        self.pc_sub = rospy.Subscriber('/point_cloud', PointCloud2, self.pc_callback, queue_size=1)
        self.img_sub = rospy.Subscriber('/image', Image, self.img_callback, queue_size=1)
        
        self.detection_pub = rospy.Publisher('/bevfusion_detections', CustomDetection3DArray, queue_size=1)
        
        self.latest_point_cloud = None
        self.latest_image = None

       
        rospy.loginfo("BEVFusion ONNX node initialized")

    def init_onnx_model(self):
        try:
            model = onnx.load(self.onnx_model_path)
            if hasattr(onnx, 'check_model'):
                onnx.check_model(model)
                rospy.loginfo("ONNX model validated successfully")
            else:
                rospy.logwarn("onnx.check_model not available, skipping model validation")
        except Exception as e:
            rospy.logerr(f"Failed to load ONNX model: {e}")
            raise RuntimeError(f"Failed to load ONNX model: {e}")

        try:
            session = ort.InferenceSession(
                self.onnx_model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                provider_options=[{'device_id': 0, 'enable_fp16': True}]
            )
        except Exception as e:
            rospy.logerr(f"Failed to initialize ONNX Runtime session: {e}")
            raise RuntimeError(f"Failed to initialize ONNX Runtime session: {e}")

        return session

    def voxelize_point_cloud(self, points: np.ndarray, voxelize_cfg: dict) -> Tuple:
        """Voxelize point cloud and prepare inputs for ONNX model."""
        voxel_size = voxelize_cfg['voxel_size']
        max_num_points = voxelize_cfg['max_num_points']
        max_voxels = voxelize_cfg['max_voxels'][0]
        point_cloud_range = voxelize_cfg['point_cloud_range']
        voxel_generator = VoxelGenerator(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels,
        )
        batch_idx = 0
        voxels_np, coordinates_np, num_points_per_voxel_np = voxel_generator.generate(points)
        
        batch_idx_array = np.zeros((coordinates_np.shape[0], 1), dtype=np.int32) + batch_idx
        coords_with_batch = np.concatenate([batch_idx_array, coordinates_np], axis=1)
        
        feats = torch.from_numpy(voxels_np).float()
        coords = torch.from_numpy(coords_with_batch).int()
        sizes = torch.from_numpy(num_points_per_voxel_np).int()
        batch_size = coords[-1, 0] + 1 if coords.shape[0] > 0 else torch.tensor(1, dtype=torch.int32)

        grid_size = [1440, 1440, 41]
        D, H, W = grid_size[2], grid_size[1], grid_size[0]
        
        num_points = torch.clamp(sizes, min=1)
        voxel_features = torch.sum(feats, dim=1) / num_points.view(-1, 1)
        
        dense_grid = torch.zeros((1, 5, D, H, W), dtype=torch.float32)
        batch_idx_t = coords[:, 0]
        z_idx = coords[:, 1]
        y_idx = coords[:, 2]
        x_idx = coords[:, 3]
        dense_grid[batch_idx_t, :, z_idx, y_idx, x_idx] = voxel_features
        
        return feats.numpy(), coords.numpy(), batch_size.numpy(), dense_grid.numpy()

    def pc_callback(self, msg: PointCloud2):
        rospy.loginfo("Received point cloud")
        points = []
        try:
            for point in pc2.read_points(msg, field_names=("x", "y", "z", "intensity", "ObjTag"), skip_nans=True):
                points.append([point[0], point[1], point[2], point[3], point[4]])
            self.latest_point_cloud = np.array(points, dtype=np.float32)
            rospy.loginfo(f"Point cloud shape: {self.latest_point_cloud.shape}")
        except Exception as e:
            rospy.logerr(f"Failed to convert point cloud: {e}")
            return
        self.process_data()

    def imgmsg_to_cv2(self, img_msg):
        if img_msg.encoding != "bgr8":
            rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
        dtype = np.dtype("uint8") # Hardcode to 8 bits...
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
        image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                        dtype=dtype, buffer=img_msg.data)

        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            image_opencv = image_opencv.byteswap().newbyteorder()
        return image_opencv

    def img_callback(self, msg: Image):
        rospy.loginfo("Received image")
        try:
            self.latest_image = self.imgmsg_to_cv2(msg)
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            return
        self.process_data()

    def process_data(self):
        if self.latest_point_cloud is None or self.latest_image is None:
            return
        
        
        try:
            bboxes_3d, scores_3d, labels_3d = self.run_inference(
                self.latest_point_cloud, 
                self.latest_image, 
                self.voxelize_cfg
            )
            self.publish_detections(bboxes_3d, scores_3d, labels_3d)
        except Exception as e:
            rospy.logerr(f"Inference failed: {e}")
        



    def run_inference(self, points: np.ndarray, image: np.ndarray, voxelize_cfg: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        expected_inputs = ['dense_grid', 'image']
        if set(self.input_names) != set(expected_inputs):
            raise ValueError(f"Model expects inputs {expected_inputs}, but got {self.input_names}")

        _, _, _, dense_grid = self.voxelize_point_cloud(points, voxelize_cfg)

        image_aug3d = ImageAug3D(final_dim=[256, 704],
                            resize_lim=[0.48, 0.48],
                            bot_pct_lim=[0.0, 0.0],
                            rot_lim=[0.0, 0.0],
                            rand_flip=False,
                            is_train=False)
        
        # Process image
        image_aug = image_aug3d(image)
        image_tensor = torch.from_numpy(image_aug).permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0)
        image_ = image_tensor.cpu().numpy()

        # Prepare inputs
        inputs = {
            self.input_names[0]: dense_grid,
            self.input_names[1]: image_
        }
        start_time = time.time()
        # Run inference
        try:
            outputs = self.session.run(self.output_names, inputs)
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")

        # End timing and calculate FPS
        end_time = time.time()
        processing_time = end_time - start_time
        print("processing_time", processing_time)
        fps = 1.0 / processing_time if processing_time > 0 else 0.0
        rospy.loginfo(f"Instantaneous FPS: {fps:.2f}")

        # Verify and process outputs
        if len(outputs) != 3:
            raise ValueError(f"Expected 3 outputs (bboxes_3d, scores_3d, labels_3d), got {len(outputs)}")
        bboxes_3d, scores_3d, labels_3d = outputs

        # Ensure output types
        bboxes_3d = np.asarray(bboxes_3d, dtype=np.float32)
        scores_3d = np.asarray(scores_3d, dtype=np.float32)
        labels_3d = np.asarray(labels_3d, dtype=np.int32)
        return bboxes_3d, scores_3d, labels_3d

    def publish_detections(self, bboxes_3d: np.ndarray, scores_3d: np.ndarray, labels_3d: np.ndarray):
        # Filter by score threshold
        mask = scores_3d > self.score_threshold
        filtered_bboxes = bboxes_3d[mask]
        filtered_scores = scores_3d[mask]
        filtered_labels = labels_3d[mask]

        print(f"Filtered Results (score > {self.score_threshold}):")
        print(f"Number of detections: {len(filtered_bboxes)}")
        for i, (bbox, score, label) in enumerate(zip(filtered_bboxes, filtered_scores, filtered_labels)):
            print(f"Detection {i+1}:")
            print(f"  BBox: {bbox}")
            print(f"  Score: {score:.4f}")
            print(f"  Label: {label}")

        # Create CustomDetection3DArray message
        detections_msg = CustomDetection3DArray()
        detections_msg.header.stamp = rospy.Time.now()
        detections_msg.header.frame_id = "base_link"  # Adjust frame_id as needed

        for bbox, score, label in zip(filtered_bboxes, filtered_scores, filtered_labels):
            detection = CustomDetection3D()
            detection.bbox = bbox.tolist()  # Store the raw bbox list
            detection.id = int(label)
            detection.score = float(score)

            # Optionally store bbox in a custom field or skip detailed bbox parsing
            detections_msg.detections.append(detection)

        self.detection_pub.publish(detections_msg)
        rospy.loginfo(f"Published {len(detections_msg.detections)} detections")

    def run(self):
        rospy.spin()

def main():
    try:
        node = BEVFusionONNXNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("BEVFusion ONNX node terminated")

if __name__ == '__main__':
    main()

