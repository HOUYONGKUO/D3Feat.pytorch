import os
import copy
import json
import torch
import numpy as np
import open3d as o3d
import torch.utils.data as data
from sklearn.neighbors import KDTree
from easydict import EasyDict as edict
from models.architectures import KPFCNN
from datasets.dataloader import get_dataloader
from utils.visualization import get_colored_point_cloud_feature


class MiniDataset(data.Dataset):
    __type__ = 'descriptor'

    def __init__(self,
                 files,
                 downsample=0.03,
                 config=None,
                 ):
        self.files = files
        self.downsample = downsample
        self.config = config

        # contrainer
        self.points = []
        self.ids_list = []
        self.num_test = 0

        for filename in files:
            pcd = o3d.io.read_point_cloud(filename)
            pcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=downsample)

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
        # pcd = o3d.io.read_point_cloud(pc_file)
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
        selected_keypoints_id = np.argsort(scores.cpu().detach().numpy(), axis=0)[::-1].squeeze()
        keypts_score = scores.cpu().detach().numpy()[selected_keypoints_id]
        keypts_loc = pts.cpu().detach().numpy()[selected_keypoints_id]
        anc_features = features.cpu().detach().numpy()[selected_keypoints_id]

        # scores = scores.detach().cpu()
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
    o3d.geometry.PointCloud.estimate_normals(source_temp)
    o3d.geometry.PointCloud.estimate_normals(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def execute_global_registration(src_keypts, tgt_keypts, src_desc, tgt_desc, distance_threshold):
    # result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    #     src_keypts, tgt_keypts, src_desc, tgt_desc, distance_threshold,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    #     4, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
    #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
    #     o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_keypts, tgt_keypts, src_desc, tgt_desc, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
         ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    # result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(src_keypts, tgt_keypts, src_desc, tgt_desc, option=None)
    return result


def refine_registration(source, target, result_ransac, distance_threshold):
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


def pcd2xyz(pcd):
    return np.asarray(pcd.points).T


def Rt2T(R, t):
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def find_mutually_nn_keypoints(ref_key, test_key, ref, test):
    """
    Use kdtree to find mutually closest keypoints

    ref_key: reference keypoints (source)
    test_key: test keypoints (target)
    ref: reference feature (source feature)
    test: test feature (target feature)
    """
    ref_features = ref.data.T
    test_features = test.data.T
    ref_keypoints = np.asarray(ref_key.points)
    test_keypoints = np.asarray(test_key.points)
    n_samples = test_features.shape[0]

    ref_tree = KDTree(ref_features)
    test_tree = KDTree(test.data.T)
    test_NN_idx = ref_tree.query(test_features, return_distance=False)
    ref_NN_idx = test_tree.query(ref_features, return_distance=False)

    # find mutually closest points
    ref_match_idx = np.nonzero(
        np.arange(n_samples) == np.squeeze(test_NN_idx[ref_NN_idx])
    )[0]
    ref_matched_keypoints = ref_keypoints[ref_match_idx]
    test_matched_keypoints = test_keypoints[ref_NN_idx[ref_match_idx]]

    return np.transpose(ref_matched_keypoints), np.transpose(test_matched_keypoints)


def compose_mat4_from_teaserpp_solution(solution):
    """
    Compose a 4-by-4 matrix from teaserpp solution
    """
    s = solution.scale
    rotR = solution.rotation
    t = solution.translation
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = rotR
    M = T.dot(R)

    if s == 1:
        M = T.dot(R)
    else:
        S = np.eye(4)
        S[0:3, 0:3] = np.diag([s, s, s])
        M = T.dot(R).dot(S)

    return M

if __name__ == '__main__':
    ######################  Preprocess  ######################
    # read pc and compute descriptors
    # read pointcloud
    point_cloud_files = ["./demo_data/cloud_bin_0.ply",
                         "./demo_data/cloud_bin_1.ply"]

    # model path
    model_pathdir = './pretrain_model/D3Feat08051813'
    # model_pathdir = './D3Feat/snapshot/D3Feat04112122'
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

    # datasets = prepare_data(point_cloud_files)

    calculate_descriptors = True
    if calculate_descriptors:
        print("###################  Calculate Descriptors  ####################")
        # calculate descriptors
        # Initiate dataset configuration
        # TODO: set downsample
        dataset = MiniDataset(files=point_cloud_files, downsample=0.03, config=config)

        dloader, _ = get_dataloader(dataset=dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=16,
                                    )
        generate_features(model.cuda(), dloader, point_cloud_files)

    ###################### Visualize  ######################
    src_pcd = o3d.io.read_point_cloud(point_cloud_files[0])
    tgt_pcd = o3d.io.read_point_cloud(point_cloud_files[1])

    src_data = np.load("./demo_data/cloud_bin_0.npz")
    src_features = o3d.pipelines.registration.Feature()
    src_features.data = src_data["features"].T
    src_keypts = o3d.geometry.PointCloud()
    src_keypts.points = o3d.utility.Vector3dVector(src_data["keypts"])
    # o3d.visualization.draw_geometries([src_keypts])  # 显示一下
    src_scores = src_data["scores"]

    tgt_data = np.load("./demo_data/cloud_bin_1.npz")
    tgt_features = o3d.pipelines.registration.Feature()
    tgt_features.data = tgt_data["features"].T
    tgt_keypts = o3d.geometry.PointCloud()
    tgt_keypts.points = o3d.utility.Vector3dVector(tgt_data["keypts"])
    # o3d.visualization.draw_geometries([tgt_keypts])
    tgt_scores = tgt_data["scores"]

    # # Show result
    # First plot the original state of the point clouds
    print("1st: plot the original state of the point clouds")
    draw_registration_result(src_pcd, tgt_pcd, np.identity(4))

    # Plot point clouds after ransac registration
    print("2nd: plot point clouds after ransac registration")
    voxel_size = 0.03
    distance_threshold = voxel_size * 1.5
    result_ransac = execute_global_registration(src_keypts, tgt_keypts, src_features, tgt_features,
                                                distance_threshold=distance_threshold)
    print(f"result_ransac: {result_ransac}")
    draw_registration_result(src_pcd, tgt_pcd, result_ransac.transformation)

    # Plot point clouds after icp registration
    print("3rd: plot point clouds after icp registration")
    radius_normal = voxel_size * 2
    src_keypts.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    tgt_keypts.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    icp_distance_threshold = voxel_size
    result_icp = refine_registration(src_keypts, tgt_keypts, result_ransac, distance_threshold=icp_distance_threshold)
    draw_registration_result(src_pcd, tgt_pcd, result_icp.transformation)
    print(f"result_icp: {result_icp}")

    ####################### Visualize the detected keypts on src_pcd and tgt_pcd #######################
    show_num = 50
    point_size = 0.03

    print("4th: Visualize the detected keypts on src_pcd")
    box_list = []
    top_k = np.argsort(src_scores, axis=0)[-show_num:]

    for i in range(show_num):
        mesh_box = o3d.geometry.TriangleMesh.create_sphere(radius=point_size)
        mesh_box.translate(src_data["keypts"][top_k[i]].reshape([3, 1]))
        mesh_box.paint_uniform_color([1, 0, 0])
        box_list.append(mesh_box)

    o3d.geometry.PointCloud.estimate_normals(src_pcd)

    src_pcd.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([src_pcd] + box_list)
    # o3d.visualization.draw_geometries_with_vertex_selection([src_pcd])

    # Visualization of Feature
    print("5th: Visualize features on src_pcd by t-SNE")
    vis_src_pcd = o3d.geometry.PointCloud()
    vis_src_pcd.points = o3d.utility.Vector3dVector(src_data["keypts"])
    o3d.geometry.PointCloud.estimate_normals(vis_src_pcd)
    vis_src_pcd = get_colored_point_cloud_feature(vis_src_pcd,
                                                  src_data["features"],
                                                  voxel_size=0.03)
    o3d.visualization.draw_geometries([vis_src_pcd])

    print("6th: Visualize the detected keypts on tgt_pcd")
    box_list = []
    top_k = np.argsort(tgt_scores, axis=0)[-show_num:]
    for i in range(show_num):
        mesh_box = o3d.geometry.TriangleMesh.create_sphere(radius=point_size)
        mesh_box.translate(tgt_data["keypts"][top_k[i]].reshape([3, 1]))
        mesh_box.paint_uniform_color([1, 0, 0])
        box_list.append(mesh_box)
    o3d.geometry.PointCloud.estimate_normals(tgt_pcd)
    tgt_pcd.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([tgt_pcd] + box_list)
    # o3d.visualization.draw_geometries_with_vertex_selection([tgt_pcd])

    # Visualization of Feature
    print("7th: Visualize features on tgt_pcd by t-SNE")
    vis_tgt_pcd = o3d.geometry.PointCloud()
    vis_tgt_pcd.points = o3d.utility.Vector3dVector(tgt_data["keypts"])
    o3d.geometry.PointCloud.estimate_normals(vis_tgt_pcd)
    vis_tgt_pcd = get_colored_point_cloud_feature(vis_tgt_pcd,
                                                  tgt_data["features"],
                                                  voxel_size=0.03)
    o3d.visualization.draw_geometries([vis_tgt_pcd])
