import os
import copy
import json
import torch
import open3d
import numpy as np
import torch.utils.data as data
from easydict import EasyDict as edict
from models.architectures import KPFCNN
from datasets.dataloader import get_dataloader


class MiniDataset(data.Dataset):
    __type__ = 'descriptor'

    def __init__(self,
                 files,
                 downsample=0.03,
                 config=None,
                 last_scene=False,
                 ):
        self.files = files
        self.downsample = downsample
        self.config = config

        # contrainer
        self.points = []
        self.ids_list = []
        self.num_test = 0

        for filename in files:
            pcd = open3d.io.read_point_cloud(filename)
            pcd = open3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=downsample)
            points = np.array(pcd.points)
            self.points += [points]
            self.ids_list += [filename]
            self.num_test += 1
        return

    def __getitem__(self, index):
        pts = self.points[index].astype(np.float32)
        feat = np.ones_like(pts[:, :1]).astype(np.float32)
        return pts, pts, feat, feat, np.array([]), np.array([])

    def __len__(self):
        return self.num_test


def generate_features(model, dloader, point_cloud_files):
    dataloader_iter = dloader.__iter__()
    for pc_file in point_cloud_files:
        # pcd = open3d.io.read_point_cloud(pc_file)
        # xyz = np.array(pcd.points)

        inputs = dataloader_iter.next()
        for k, v in inputs.items():  # load inputs to device.
            if type(v) == list:
                inputs[k] = [item.cuda() for item in v]
            else:
                inputs[k] = v.cuda()
        features, scores = model(inputs)
        pcd_size = inputs['stack_lengths'][0][0]
        pts = inputs['points'][0][:int(pcd_size)]
        features, scores = features[:int(pcd_size)], scores[:int(pcd_size)]

        # selecet keypoint based on scores
        # scores_first_pcd = scores[inputs['in_batches'][0][:-1]]
        # select_num = 1000
        # selected_keypoints_id = np.argsort(scores.cpu().detach().numpy(), axis=0)[:].squeeze()[0:select_num]

        selected_keypoints_id = np.argsort(scores.cpu().detach().numpy(), axis=0)[:].squeeze()
        keypts_score = scores.cpu().detach().numpy()[selected_keypoints_id]
        keypts_loc = pts.cpu().detach().numpy()[selected_keypoints_id]
        anc_features = features.cpu().detach().numpy()[selected_keypoints_id]

        # save keypts/features/scores in ".npz" format
        np.savez_compressed(
            pc_file.replace(".ply", ""),
            keypts=keypts_loc.astype(np.float64),
            features=anc_features.astype(np.float64),
            scores=keypts_score.astype(np.float64),
        )


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    open3d.geometry.PointCloud.estimate_normals(source_temp)
    open3d.geometry.PointCloud.estimate_normals(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    open3d.visualization.draw_geometries([source_temp, target_temp])


def execute_global_registration(src_keypts, tgt_keypts, src_desc, tgt_desc, distance_threshold):
    result = open3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_keypts, tgt_keypts, src_desc, tgt_desc, True, distance_threshold,
        open3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [open3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
         ], open3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def refine_registration(source, target, result_ransac, distance_threshold):
    result = open3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        open3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


if __name__ == '__main__':
    ######################  Preprocess  ######################
    # read pc and compute descriptors
    # read pointcloud
    point_cloud_files = ["./demo_data/cloud_bin_0.ply",
                         "./demo_data/cloud_bin_1.ply"]

    # model path
    # TODO: choose your model path
    model_pathdir = 'D3Feat/snapshot/D3Feat04112122'
    config_path = os.path.join(model_pathdir, 'config.json')
    model_path = os.path.join(model_pathdir, 'models/model_best_acc.pth')
    save_path = ["./demo_data/cloud_bin_0.npz",
                 "./demo_data/cloud_bin_1.npz"]

    config = json.load(open(config_path, 'r'))
    config = edict(config)

    # create model
    config.architecture = [
        'simple',
        'resnetb',
    ]
    for i in range(config.num_layers - 1):
        config.architecture.append('resnetb_strided')
        config.architecture.append('resnetb')
        config.architecture.append('resnetb')
    for i in range(config.num_layers - 2):
        config.architecture.append('nearest_upsample')
        config.architecture.append('unary')
    config.architecture.append('nearest_upsample')
    config.architecture.append('last_unary')

    model = KPFCNN(config)

    model.load_state_dict(torch.load(model_path)['state_dict'], strict=False)
    print(f"Load weight from {model_path}")
    model.eval()

    # TODO: choose whether calculate descriptors
    calculate_descriptors = True
    if calculate_descriptors == True:
        # calculate descriptors
        # Initiate dataset configuration
        dataset = MiniDataset(files=point_cloud_files, downsample=0.03, config=config)

        dloader, _ = get_dataloader(dataset=dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=16,
                                    )
        generate_features(model.cuda(), dloader, point_cloud_files)

    ###################### Visualize  ######################
    # show pc and registration
    # Load the descriptors and estimate the transformation parameters using RANSAC
    src_pcd = open3d.io.read_point_cloud(point_cloud_files[0])
    tgt_pcd = open3d.io.read_point_cloud(point_cloud_files[1])

    # TODO: define src descriptors file path
    src_data = np.load(save_path[0])
    src_features = open3d.pipelines.registration.Feature()
    src_features.data = src_data["features"].T
    src_keypts = open3d.geometry.PointCloud()
    src_keypts.points = open3d.utility.Vector3dVector(src_data["keypts"])
    src_scores = src_data["scores"]

    # TODO: define tgt descriptors file path
    tgt_data = np.load(save_path[1])
    tgt_features = open3d.pipelines.registration.Feature()
    tgt_features.data = tgt_data["features"].T
    tgt_keypts = open3d.geometry.PointCloud()
    tgt_keypts.points = open3d.utility.Vector3dVector(tgt_data["keypts"])
    tgt_scores = tgt_data["scores"]

    # Show result
    # First plot the original state of the point clouds
    draw_registration_result(src_pcd, tgt_pcd, np.identity(4))

    # Plot point clouds after ransac registration
    # TODO: set voxel_size/distance_threshold/radius_normal/icp_distance_threshold
    voxel_size = 0.03
    distance_threshold = voxel_size * 1.5
    result_ransac = execute_global_registration(src_keypts, tgt_keypts, src_features, tgt_features, distance_threshold=distance_threshold)
    print(f"result_ransac: {result_ransac}")
    draw_registration_result(src_pcd, tgt_pcd, result_ransac.transformation)

    # Plot point clouds after icp registration
    radius_normal = voxel_size * 2
    src_keypts.estimate_normals(open3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    tgt_keypts.estimate_normals(open3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    icp_distance_threshold = voxel_size
    result_icp = refine_registration(src_keypts, tgt_keypts, result_ransac, distance_threshold=icp_distance_threshold)
    draw_registration_result(src_pcd, tgt_pcd, result_icp.transformation)
    print(f"result_icp: {result_icp}")

    # Visualize the detected keypts on src_pcd and tgt_pcd
    # TODO: set show keypts num and keypts size
    show_num = 50
    keypts_radius = 0.03
    box_list = []
    top_k = np.argsort(tgt_scores, axis=0)[-show_num:]
    for i in range(show_num):
        mesh_box = open3d.geometry.TriangleMesh.create_sphere(radius=keypts_radius)
        mesh_box.translate(tgt_data["keypts"][top_k[i]].reshape([3, 1]))
        mesh_box.paint_uniform_color([1, 0, 0])
        box_list.append(mesh_box)

    open3d.geometry.PointCloud.estimate_normals(tgt_pcd)
    tgt_pcd.paint_uniform_color([0, 0.651, 0.929])
    open3d.visualization.draw_geometries([tgt_pcd] + box_list)
    
    # visual feature
    import torch.nn as nn
    Linear = nn.Linear(32, 3)
    feat_3d = Linear(torch.tensor(tgt_data["features"]).float())
    feat_3d = normal(feat_3d)
    tgt_keypts.colors = open3d.utility.Vector3dVector(feat_3d.detach().numpy())
    open3d.visualization.draw_geometries([tgt_keypts])


    box_list = []
    top_k = np.argsort(src_scores, axis=0)[-show_num:]
    for i in range(show_num):
        mesh_box = open3d.geometry.TriangleMesh.create_sphere(radius=keypts_radius)
        mesh_box.translate(src_data["keypts"][top_k[i]].reshape([3, 1]))
        mesh_box.paint_uniform_color([1, 0, 0])
        box_list.append(mesh_box)

    open3d.geometry.PointCloud.estimate_normals(src_pcd)
    src_pcd.paint_uniform_color([1, 0.706, 0])
    open3d.visualization.draw_geometries([src_pcd] + box_list)
    
    # visual feature
    Linear = nn.Linear(32, 3)
    feat_3d = Linear(torch.tensor(src_data["features"]).float())
    feat_3d = normal(feat_3d)
    src_keypts.colors = open3d.utility.Vector3dVector(feat_3d.detach().numpy())
    open3d.visualization.draw_geometries([src_keypts])
