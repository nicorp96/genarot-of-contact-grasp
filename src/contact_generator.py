import open3d as o3d
import numpy as np
import logging
import configparser
import os
from src.point_cloud_utils import (
    get_contact_grasp_points_o3d,
    sampling,
    read_mesh,
    get_orths,
    to_trimesh,
)


class ContactGenerator:
    def __init__(self, config: configparser.ConfigParser(), log_level) -> None:
        self._config = config
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)
        self._mesh_o3d = None
        self._tmesh = None

    def set_meshes(self, mesh_path):
        self._mesh_o3d = read_mesh(mesh_path)
        self._tmesh = to_trimesh(self._mesh_o3d)

    @staticmethod
    def select_points_in_contact(vertices, normals, color_n, threshold, n_desired):
        n_points = 0  # Init: number of found points.
        ndcs = []

        while n_points <= n_desired and np.round(threshold, 2) >= 0.6:
            ndcs = (
                color_n >= threshold
            )  # Selection of the number of points in fingerprints (value of normalized color value > threshold).
            n_points = np.sum(ndcs)  # Number of found points.
            threshold -= 0.0001  # If the found number is below the desired one, start another run with a lower threshold.

        vertices = vertices[ndcs]  # Vertices of the selected points.
        dirs = normals[ndcs]  # Vertex normals of the selected points.
        return vertices, dirs

    @staticmethod
    def preprocess_colors(colors):
        colors_3 = (
            colors[:, :3] / 255
        )  # Preprocessing: color values between 0 and 1, 4th column will be removed.
        colors_n = np.linalg.norm(colors_3, axis=1) / np.sqrt(
            3
        )  # Normalization of the three color values to a value between 0 and 1.
        return colors_n

    @staticmethod
    def compute_rays_of_contact_points(orgs, dirs, alpha, n_samplepoints):
        # Compute fans of rays consisting of a sampled cone.
        c = np.tan(alpha)
        angs = np.linspace(0, 2 * np.pi, n_samplepoints, endpoint=False)
        p, d = [], []
        for org, dir in zip(orgs, dirs):
            a = dir / np.linalg.norm(dir)
            n1, n2 = get_orths(a)
            for ang in angs:
                p.append(org)
                p.append(org)
                d.append(a + c * np.cos(ang) * n1 + c * np.sin(ang) * n2)
                d.append(-a - c * np.cos(ang) * n1 - c * np.sin(ang) * n2)
        return p, d

    @staticmethod
    def get_intersect_location_pairs(tmesh, ray_origins, ray_directions):
        # Returns unique cartesian locations where rays hit the mesh.
        locations, index_ray, index_tri = tmesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_directions
        )

        contact_points_1 = []
        contact_points_2 = []

        for ray_ndx in range(len(ray_origins)):
            orig = ray_origins[ray_ndx]
            res_ndcs = np.array(
                [k for k in range(len(index_ray)) if index_ray[k] == ray_ndx]
            )
            # Less than two intersections shows that the line is pointing away from the object, so it is discarded.
            if len(res_ndcs) < 2:
                continue
            dist = [np.linalg.norm(loc - orig) for loc in locations[res_ndcs]]
            res_ndcs_sorted = res_ndcs[np.argsort(dist)]

            from_ndx, to_ndx = res_ndcs_sorted[0], res_ndcs_sorted[-1]
            # Double wall extension.
            # If there is an even number of points with the mesh greater than 2, an additional end point is chosen for the shortest distance.

            if len(res_ndcs) > 2 and len(res_ndcs) % 2 == 0:
                from_ndx, to_ndx = res_ndcs_sorted[0], res_ndcs_sorted[1]

            contact_points_1.append(locations[from_ndx])
            contact_points_2.append(locations[to_ndx])

        contact_points_1 = np.vstack(contact_points_1)
        contact_points_2 = np.vstack(contact_points_2)
        return contact_points_1, contact_points_2

    def find_contact_grasp(self):
        # Identification of the starting points.
        mesh_colors = np.asarray(self._tmesh.visual.vertex_colors)
        mesh_vertices = np.asarray(self._tmesh.vertices)
        mesh_normals = np.asarray(self._tmesh.vertex_normals)
        mesh_colors_n = self.preprocess_colors(mesh_colors)

        self._logger.debug("Selecting contact points in 3D Mesh")
        (contact_origins, contact_directions,) = self.select_points_in_contact(
            mesh_vertices,
            mesh_normals,
            mesh_colors_n,
            self._config["contact_grasp"].getfloat("begin_point_threshold"),
            self._config["contact_grasp"].getint("n_lines_points"),
        )
        if contact_origins.shape[0] > self._config["contact_grasp"].getint(
            "max_num_points_in_contact"
        ):
            contact_origins, contact_directions = sampling(
                contact_origins,
                contact_directions,
                num_points=self._config["contact_grasp"].getint(
                    "num_contact_points_sampling"
                ),
            )
            self._logger.debug(f"Sampling from {contact_origins.shape[0]}")
        self._logger.debug("Computing rays of selected contact points")
        alphas = [
            self._config["contact_grasp"].getfloat("alpha_1"),
            self._config["contact_grasp"].getfloat("alpha_2"),
            # self._config["contact_grasp"].getfloat("alpha_3"),
        ]

        (
            ray_contact_points,
            ray_contact_points_dirs,
        ) = self.get_rays_origin(alphas, contact_origins, contact_directions)
        ray_contact_points = np.stack(ray_contact_points)
        ray_contact_points_dirs = np.stack(ray_contact_points_dirs)

        contact_points_1, contact_points_2 = self.get_intersect_location_pairs(
            self._tmesh,
            ray_contact_points,
            ray_contact_points_dirs,
        )

        vertices_contact, vertices_normal, colors = get_contact_grasp_points_o3d(
            self._mesh_o3d
        )
        vertices_contact_down, _, colors = sampling(
            vertices_contact,
            vertices_normal,
            colors,
            num_points=self._config["contact_grasp"].getint("num_points_sampling"),
        )
        return contact_points_1, contact_points_2, vertices_contact_down, colors

    def get_rays_origin(self, alphas, contact_origins, contact_directions):
        ray_contact_points = []
        ray_contact_points_dirs = []
        for alpha in alphas:
            ray_points, ray_directions = self.compute_rays_of_contact_points(
                contact_origins,
                contact_directions,
                alpha,
                self._config["contact_grasp"].getint("n_lines_points"),
            )
            ray_contact_points += ray_points
            ray_contact_points_dirs += ray_directions
        return ray_contact_points, ray_contact_points_dirs

    @staticmethod
    def compute_points(
        contact_points_1, contact_points_2, mesh, contact_grasp, colors, num_points=6000
    ):
        pcd_1 = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(contact_points_1)
        )
        pcd_2 = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(contact_points_2)
        )

        pcd_1.paint_uniform_color([0.70196078, 0.70196078, 0.70196078])
        pcd_2.paint_uniform_color([0.70196078, 0.70196078, 0.70196078])
        total_points = (
            num_points
            - contact_grasp.shape[0]
            - contact_points_1.shape[0]
            - contact_points_2.shape[0]
        )
        pcd_3 = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(contact_grasp)
        )
        pcd_3.colors = o3d.utility.Vector3dVector(colors)
        point_cloud = mesh.sample_points_uniformly(number_of_points=15000)
        point_cloud = mesh.sample_points_poisson_disk(
            number_of_points=total_points, pcl=point_cloud
        )
        pc_complete = point_cloud + pcd_1 + pcd_2 + pcd_3
        points = np.asarray(pc_complete.points)
        colors = np.asarray(pc_complete.colors)
        pc_complete.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        normals = np.asarray(pc_complete.normals)
        points_normals = np.append(points, normals, 1)
        return points_normals, colors

    @staticmethod
    def get_label_from_color(num_points, colors, axis_color):
        label = np.zeros(num_points)
        idx = colors[:, axis_color] == 1.0
        label[idx] = 1.0
        label = np.expand_dims(label, axis=0)
        return label

    @staticmethod
    def numpy_save(name, folder_name, points, contact_grasp_points, label, colors):
        object_name = folder_name + "_" + name
        path = os.path.join(os.getcwd(), "data_pcl_contact", object_name)
        np.savez(
            path,
            points=points,
            contact_grasp_points=contact_grasp_points,
            label=label,
            colors=colors,
        )

    @staticmethod
    def show_all_contact_points(
        points_normals, label_contact_points, label_grasp_points
    ):
        idx = label_contact_points[0, :] == 1.0
        point_cloud_contact_points = points_normals[:, :3][idx]
        point_cloud_cp = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(point_cloud_contact_points)
        )
        point_cloud_cp.paint_uniform_color([1, 0, 0])

        idx = label_grasp_points[0, :] == 1.0
        point_cloud_contact_points = points_normals[:, :3][idx]
        point_cloud_gp = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(point_cloud_contact_points)
        )
        point_cloud_gp.paint_uniform_color([0, 0, 1])

        point_cloud = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(points_normals[:, :3])
        )
        point_cloud.paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries([point_cloud, point_cloud_gp, point_cloud_cp])

        print(points_normals.shape)
        print(label_grasp_points.shape)
        print(label_contact_points.shape)
