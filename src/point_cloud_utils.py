import numpy as np
import os
import open3d as o3d
import trimesh
import torch


def get_mesh_filename(path):
    # session_name = "full{:s}_{:s}".format(session_number, instruction)
    # base_dir = os.path.join(data_dirs[int(session_number) - 1], session_name)
    object_name_filename = os.path.join(path, "object_name.txt")
    with open(object_name_filename, "r") as f:
        object_name = f.readline().strip()
    return os.path.join(
        path,
        "thermal_images",
        "{:s}_textured{:s}.ply".format(object_name, ""),
    )


def read_mesh(fn, format="o3d"):
    if format == "o3d":
        m = o3d.io.read_triangle_mesh(fn)
        if not m.has_vertex_normals():
            m.compute_vertex_normals()
            m.compute_triangle_normals()
    elif format == "trimesh":
        m = trimesh.load(fn)
    return m


def to_trimesh(mesh):
    mesh.compute_adjacency_list()
    tmesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.triangles,
        face_normals=mesh.triangle_normals,
        vertex_normals=mesh.vertex_normals,
        vertex_colors=mesh.vertex_colors,
        use_embree=True,
    )
    return tmesh


def get_orths(v):  # v = ray directions (vertex normal)
    # v = v / np.linalg.norm(v)
    k = np.argmax(
        np.abs(v)
    )  # Index des größten Wertes (Betrag!) im Array wird zurückgeliefert: [-0.833 -0.546 0.086] liefert 0 zurück
    # setze die restlichen Koordinaten auf beliebigen Wert!=0, etwa 1 0
    if k == 0:  # points in x-direction
        x = np.array([-v[1] / v[0], 1.0, 0.0])
    elif k == 1:  # points in y-direction
        x = np.array([1, -v[0] / v[1], 0.0])
    elif k == 2:  # points in z-direction
        x = np.array([1.0, 0.0, -v[0] / v[2]])
    x = x / np.linalg.norm(x)
    x2 = np.cross(v, x)
    return x, x2


def numpy_save(name, folder_name, points, contact_grasp, label, colors):
    object_name = folder_name + "_" + name
    path = os.path.join(os.getcwd(), "data_pcl_contact", object_name)
    np.savez(
        path, points=points, contact_grasp=contact_grasp, label=label, colors=colors
    )


def group_points(points, idx):
    device = points.device
    batch = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_idx = (
        torch.arange(batch, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_idx, idx, :]
    return new_points


def farthest_point_sample(x_y_z, n_points):
    device = x_y_z.device
    batch = x_y_z.shape[0]
    n_ds = x_y_z.shape[1]
    centroids = torch.zeros(batch, n_points, dtype=torch.long).to(device)
    distance = torch.ones(batch, n_ds, dtype=torch.float64).to(device) * 1e10
    farthest = torch.randint(0, n_ds, (batch,), dtype=torch.long).to(device)
    batch_indices = torch.arange(batch, dtype=torch.long).to(device)
    for i in range(n_points):
        centroids[:, i] = farthest
        centroid = x_y_z[batch_indices, farthest, :].view(batch, 1, 3)
        dist = torch.sum((x_y_z - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def sampling(points, normals, colors=None, num_points=1000):
    x_y_z = torch.from_numpy(points)
    x_y_z = torch.unsqueeze(x_y_z, dim=0)
    n_p = torch.from_numpy(normals)
    n_p = torch.unsqueeze(n_p, dim=0)
    fsp_idx = farthest_point_sample(x_y_z, num_points)
    new_x_y_z = group_points(x_y_z, fsp_idx)
    new_normals = group_points(n_p, fsp_idx)
    new_x_y_z = new_x_y_z.cpu().numpy().squeeze()
    new_normals = new_normals.cpu().numpy().squeeze()
    if colors is not None:
        n_c = torch.from_numpy(colors)
        n_c = torch.unsqueeze(n_c, dim=0)
        new_colors = group_points(n_c, fsp_idx)
        new_colors = new_colors.cpu().numpy().squeeze()
        return new_x_y_z, new_normals, new_colors
    return new_x_y_z, new_normals


def get_contact_grasp_points_o3d(m):
    v_idx = 0
    th = 0.5
    c = np.asarray(m.vertex_colors)  # Colors of all corner points.
    v = np.asarray(m.vertices)  # All vertices.
    n = np.asarray(m.vertex_normals)  # All vertex normals.
    print(c.min())
    print(c.max())
    while v_idx < 1001 or v_idx > 5000:
        idx = c > th
        vertices_contact = v[idx]
        normals_contact = n[idx]
        v_idx = vertices_contact.shape[0]
        if v_idx < 1001:
            th = th - 0.001
        else:
            th = th + 0.001
    vertices_contact = vertices_contact.reshape(-1, 3)
    normals_contact = normals_contact.reshape(-1, 3)
    colors = c[idx]
    colors = colors.reshape(-1, 3)
    return vertices_contact, normals_contact, colors


def get_contact_grasp_from_point_cloud(points, colors):
    v_idx = 0
    th = 0.4
    # while v_idx < 1000 or v_idx > 5000:
    colors = np.tile(colors, (3, 1)).T
    points = points.T
    new_points = []
    new_colors = []
    print(colors.shape)
    print(points.shape)
    for i in range(points.shape[0]):
        if colors[i, :].all() < 0.6:
            new_points.append(points[i, :])
            new_colors.append(colors[i, :])

    print(new_points[0])
    new_points = np.vstack(new_points)
    new_colors = np.vstack(new_colors)
    print(new_points.shape)
    print(new_colors.shape)
    # vertices_contact = points[colors > th]
    # normals_contact = n[idx]
    # v_idx = vertices_contact.shape[0]
    #     if v_idx < 1000:
    #         th -= 0.1
    #     else:
    #         th += 0.1
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(new_points))
    pcd.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([pcd])
    # o3d.visualization.draw_geometries([pcl, pcd])
    return vertices_contact


def get_pcl_contact_points(o3d_mesh, contact_grasp, contact_normals):
    pcd_3 = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(contact_grasp))
    pcd_3.normals = o3d.utility.Vector3dVector(contact_normals)
    pcd_3.paint_uniform_color([1, 0, 0])
    point_cloud = o3d_mesh.sample_points_uniformly(number_of_points=10000)
    point_cloud = o3d_mesh.sample_points_poisson_disk(
        number_of_points=3000 - contact_grasp.shape[0], pcl=point_cloud
    )
    # point_cloud = o3d_mesh.sample_points_poisson_disk(
    #     number_of_points=10000, pcl=point_cloud
    # )
    point_cloud.paint_uniform_color([0, 0, 0])
    pc_complete = point_cloud + pcd_3

    # o3d.visualization.draw_geometries([pcd_3])
    # o3d.visualization.draw_geometries([pc_complete])
    points = np.asarray(pc_complete.points)
    colors = np.asarray(pc_complete.colors)
    pc_complete.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    normals = np.asarray(pc_complete.normals)
    points_normals = np.append(points, normals, 1)
    # pcd = o3d.geometry.PointCloud(
    #     points=o3d.utility.Vector3dVector(points_normals[:, :3])
    # )
    # o3d.visualization.draw_geometries([pc_complete])
    label = np.zeros(points.shape[0])
    idx = colors[:, 0] == 1.0
    label[idx] = 1.0
    label = np.expand_dims(label, axis=0)

    point_cloud_true = points_normals[:, :3][idx]
    point_cloud_t = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(point_cloud_true)
    )

    point_cloud_t.paint_uniform_color([0, 1, 0])
    # o3d.visualization.draw_geometries([point_cloud_t, point_cloud])
    print(points_normals.shape)
    print(colors.shape)
    print(contact_grasp.shape)
    return points_normals, contact_grasp, label, colors


def get_point_cloud_from_numpy(file_path):
    x, y, z, c, xx, yy, zz = np.load(file_path)
    pts = np.vstack((xx, yy, zz))
    print(f"shape of points = {pts.shape}")
    offset = (pts.max(1, keepdims=True) + pts.min(1, keepdims=True)) / 2
    pts -= offset
    scale = max(pts.max(1) - pts.min(1)) / 2
    pts /= scale
    pts_choice = np.random.choice(pts.shape[1], size=10000, replace=True)
    pts = pts[:, pts_choice]
    colors = c[pts_choice]
    # colors = np.tile(c, (3, 1)).T
    # point_cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts.T))
    # point_cloud.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([point_cloud])
    return pts, colors


def generate_random_pcl(mesh):
    choice = np.random.choice(4, 1, replace=False)
    if choice == 1:
        geometry = o3d.geometry.TriangleMesh.create_icosahedron(radius=0.02)
    elif choice == 2:
        geometry = o3d.geometry.TriangleMesh.create_cylinder(radius=0.02, height=0.02)
    elif choice == 3:
        geometry = o3d.geometry.TriangleMesh.create_cone(radius=0.02, height=0.02)
    else:
        geometry = o3d.geometry.TriangleMesh.create_box(
            width=0.02, height=0.02, depth=1.0
        )

    point_cloud = geometry.sample_points_uniformly(number_of_points=20000)
    points = np.asarray(point_cloud.points)
    point_cloud.translate(mesh.get_center())
    return points


if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), "data")
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(data_dir, folder_name, file_name)
            fn = get_mesh_filename(file_path)
            m = read_mesh(fn)
            tmesh = to_trimesh(m)
            vertices_contact, normals_contact = get_contact_grasp_points_o3d(m)
            # print(vertices_contact.shape)
            # vertices_contact, normals_contact = get_contact_grasp_points(tmesh)
            # print(vertices_contact.shape)
            points, contact_grasp, label, colors = get_pcl_contact_points(
                m, sampling(vertices_contact), normals_contact
            )
            numpy_save(file_name, folder_name, points, contact_grasp, label, colors)
            # points = generate_random_pcl(m)
            # numpy_save(file_name + "-6", folder_name, points, contact_grasp, label)

    # data_dir = os.path.join(os.getcwd(), "data_pcl_contact")
    # labels = []
    # counter = True
    # # label_total = np.ones(10)
    # for folder_name in os.listdir(data_dir):
    #     if "pan" in folder_name:
    #         folder_path = os.path.join(data_dir, folder_name)
    #         array = np.load(folder_path)
    #         points = array["points"]
    #         label = array["label"]
    #         print(folder_name)
    #         print(label.shape)
    #         if counter:
    #             label_total = label[0, :]
    #             print(label_total.shape)
    #         else:
    #             label_total = np.concatenate(label_total, label[0, :])
    #         counter = False
    #         idx = label_total == 1.0
    #         point_c = points[:, :3][idx]
    #         print(point_c.shape)
