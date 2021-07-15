
from lib.utils_vvd.image_processing import image_resize
from .mmcontrollers.classifier import Controller as Classifier
import numpy as np
from .utils_vvd import vvd_ceil
from .utils_vvd import plt_image_show
from .utils_vvd import vvd_round
from .utils_vvd import crop_by_cycle_y_min_max
import cv2


class Controller(object):
    def __init__(self, checkpoint, config_file, **kwargs):
        self._classifier = Classifier(checkpoint, config_file, **kwargs)
        self._classnames = [
            'OK', 'NG'
        ]

    @staticmethod
    def get_center(circle_info):
        return np.mean(np.array(circle_info), axis=0)[:2].tolist()

    @staticmethod
    def source_polar_transfer(image, source_center_x_y, source_radius_info):

        source_focus_radius_min, source_focus_radius_max = source_radius_info

        assert source_focus_radius_max > source_focus_radius_min

        polar_height = vvd_ceil(source_focus_radius_max * 2 * np.pi)
        polar_width = vvd_ceil(source_focus_radius_max)

        source_center_x_y = vvd_round(source_center_x_y)

        source_polar = cv2.warpPolar(image, (polar_width, polar_height), source_center_x_y, source_focus_radius_max, 12)

        source_polar = (source_polar[::-1, ...]).astype('uint8')

        return source_polar

    @staticmethod
    def get_phase_crop(source_polar, cycle_angle, center_phase, radius_min):

        y_min = vvd_round((center_phase - cycle_angle/2) % 360 / 360 * source_polar.shape[0])
        y_max = vvd_round((center_phase + cycle_angle/2) % 360 / 360 * source_polar.shape[0])

        data_crop = crop_by_cycle_y_min_max(source_polar, y_min, y_max)

        data_crop = data_crop[:, vvd_round(radius_min):]

        return data_crop

    def get_crop_list(self, source_polar, cycle_angle, phase_list, radius_min):
        crop_list = list()
        for center_phase in phase_list:
            data_crop = self.get_phase_crop(source_polar, cycle_angle, center_phase, radius_min)
            crop_image = np.asfortranarray(data_crop.transpose(1, 0, 2)[::-1, ...].astype('float32'))
            crop_list.append(crop_image)
        return crop_list

    @staticmethod
    def input_resize(image, circle_info, resize_factor):
        resized_image = image_resize(image, factor=resize_factor)

        def resize_circle_info(circle_info):
            resized_circle_info = list()
            for circle_data in circle_info:
                sub_data = list()
                for data in circle_data:
                    sub_data.append(data * resize_factor)
                resized_circle_info.append(tuple(sub_data))
            return resized_circle_info

        resized_circle_info = resize_circle_info(circle_info)
        return resized_image, resized_circle_info

    def infer(self, image, circle_info, carrier_info, **kwargs):
        """
        :image: giant image to query
        :circle_info: circle_info of input image
        :carrier_info: output of Vi_cA_13_II
        """

        target_carrier_width = 160
        current_carrier_width = circle_info[2][-1] - circle_info[1][-1]
        carrier_width = circle_info[2][-1] - circle_info[1][-1]

        if current_carrier_width > target_carrier_width:
            resize_factor = target_carrier_width / max(1, current_carrier_width)
            resized_image, resized_circle_info = self.input_resize(image, circle_info, resize_factor)
        else:
            resized_circle_info = circle_info
            resized_image = image

        source_radius_info = [resized_circle_info[1][-1], resized_circle_info[2][-1]]
        center_xy = self.get_center(resized_circle_info)

        source_polar = self.source_polar_transfer(resized_image, center_xy, source_radius_info)

        phase_list = carrier_info['ball_phase_list'] + carrier_info['revit_phase_list']
        point_list = carrier_info['ball_point_list'] + carrier_info['revit_point_list']

        cycle_angle = carrier_info['cycle_angle']

        crop_list = self.get_crop_list(source_polar, cycle_angle, phase_list, resized_circle_info[1][-1])

        distributions = self._classifier(crop_list)

        assert len(distributions) == len(crop_list) == len(point_list) == len(phase_list)

        box_size = 0.3

        results = list()
        for distribution, center_point in zip(distributions, point_list):
            assert len(distribution) == len(self._classnames)
            x, y = center_point
            bbox = vvd_round([x - carrier_width * box_size, y - carrier_width * box_size, x + carrier_width * box_size, y + carrier_width * box_size])
            results.append(
                dict(
                    scores = distribution,
                    classnames = self._classnames,
                    bbox = bbox
                )
            )
        ng_results = list()
        for result in results:
            score = result['scores']
            classname = result['classnames']
            bbox = result['bbox']

            if score[1] > 0.8:
                ng_results.append(bbox)

        return ng_results

    __call__ = infer
