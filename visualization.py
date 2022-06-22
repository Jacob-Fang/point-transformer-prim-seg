import h5py
import numpy as np
import open3d as o3d

if __name__ == '__main__':

    with h5py.File('/home/fz20/Project/point-transformer-prim-seg/exp/ABC/pt_3layer/test/visualization/8780.h5', "r") as hf:

            print(list(hf.keys()))
            # N x 3
            test_points = np.array(hf.get("points"))
            test_points = np.transpose(test_points, (0, 1, 3, 2))
            print(test_points.shape)

            # N x 1
            test_labels = np.array(hf.get("seg_gt"))
            print(test_labels.shape)

            # N x 3
            test_pred_primitives = np.array(hf.get("seg_pre"))
            print(test_pred_primitives.shape)

            # test_pred_bound = np.array(hf.get("bound_pre"))

            # test_bound = np.array(hf.get("bound_gt"))

            # print(test_points)
            # print(test_points.shape[0],test_points.shape[1],test_points.shape[2])

    colors = np.random.rand(10000, 3)
    bound_color = np.array([[0.41176,0.41176,0.41176], [1,0,0]])
    for i in range(len(test_points)):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(test_points[i][0])
        c = []
        for j in range(len(test_points[i][0])):
            c.append(colors[test_pred_primitives[i][0][j]])
        c = np.asarray(c)
        pcd.colors = o3d.utility.Vector3dVector(c)
        o3d.io.write_point_cloud('/home/fz20/Project/point-transformer-prim-seg/exp/ABC/pt_3layer/test/visualization/{}.ply'.format(str(i+1) + "_seg_pre"), pcd)

        # c = []
        # for j in range(len(test_points[i][0])):
        #     c.append(bound_color[test_pred_bound[i][0][j]])
        # c = np.asarray(c)
        # pcd.colors = o3d.utility.Vector3dVector(c)
        # o3d.io.write_point_cloud('/home/fz20/Project/Boundary_prim_seg/visualization/4.18newdataset//{}.ply'.format(str(i+1) + "_boundary_pre"), pcd)
        # c = []
        # for j in range(len(test_points[i][0])):
        #     c.append(bound_color[test_bound[i][0][j]])
        # c = np.asarray(c)
        # pcd.colors = o3d.utility.Vector3dVector(c)
        # o3d.io.write_point_cloud('/home/fz20/Project/Boundary_prim_seg/visualization/4.18newdataset//{}.ply'.format(str(i+1) + "_boundary_gt"), pcd)

        c = []
        for j in range(len(test_points[i][0])):
            c.append(colors[test_labels[i][0][j]])
        c = np.asarray(c)
        pcd.colors = o3d.utility.Vector3dVector(c)
        o3d.io.write_point_cloud('/home/fz20/Project/point-transformer-prim-seg/exp/ABC/pt_3layer/test/visualization/{}.ply'.format(str(i+1) + "_seg_gt"), pcd)