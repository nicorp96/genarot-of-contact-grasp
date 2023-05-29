import logging
import configparser
import json
import os
from src.contact_generator import ContactGenerator
from src.point_cloud_utils import get_mesh_filename


class MainClass:
    def __init__(self, path: str, log_level) -> None:
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)
        self._config = configparser.ConfigParser()
        self._config.read_dict(self._open_json(path))
        self._contact_generator = ContactGenerator(
            self._config,
            log_level,
        )

    @staticmethod
    def _open_json(config_path):
        with open(config_path) as config_file:
            data = json.loads(config_file.read())
        return data

    def run(self):
        self._logger.debug("Generating data for neural network")

        data_dir = os.path.join(os.getcwd(), "data")
        for folder_name in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder_name)
            # if "full1" not in folder_path:
            #     continue
            for file_name in os.listdir(folder_path):
                # if "mug" not in file_name:
                #     continue
                file_path = os.path.join(data_dir, folder_name, file_name)
                self._logger.debug("folder session: " + folder_name)
                self._logger.debug("object name: " + file_name)
                mesh_path = get_mesh_filename(file_path)
                self._contact_generator.set_meshes(mesh_path)
                (
                    c1,
                    c2,
                    vertices_contact,
                    colors,
                ) = self._contact_generator.find_contact_grasp()
                points_normals, colors = self._contact_generator.compute_points(
                    c1,
                    c2,
                    self._contact_generator._mesh_o3d,
                    vertices_contact,
                    colors,
                    num_points=self._config["contact_grasp"].getint("num_points_x"),
                )
                label_contact_points = self._contact_generator.get_label_from_color(
                    points_normals.shape[0], colors, 0
                )
                label_grasp_points = self._contact_generator.get_label_from_color(
                    points_normals.shape[0], colors, 2
                )

                self._contact_generator.numpy_save(
                    file_name,
                    folder_name,
                    points_normals,
                    label_grasp_points,
                    label_contact_points,
                    colors,
                )
            # self._contact_generator.show_all_contact_points(
            #     points_normals, label_contact_points, label_grasp_points
            # )
