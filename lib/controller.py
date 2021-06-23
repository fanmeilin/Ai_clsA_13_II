
from .mmcontrollers.classifier import Controller as Classifier
import numpy as np

class Controller(object):
    def __init__(self, checkpoint, config_file, **kwargs):
        self._classifier = Classifier(checkpoint, config_file, **kwargs)
        self._classnames = [
            'NG', 'Foreign_object', 'Partical', 'Shenxi', 'Scratch', 'Tou_residue', 
            'Zhu_residue', 'Metal_residue', 'Bump_class', 'Bump_missing', 'Bump_blur', 
            'Bump_deformed', 'Bump_foreign', 'Bump_pushed', 'Bump_large', 'Bump_small', 
            'Wire_class', 'Wire_short', 'Wire_open', 'Wire_Oxidize', 'Wire_Defect', 
            'PI_class', 'PI_defect', 'PI_bubble', 'PI_hole', 'PI_black_dot', 
            'IQC_class', 'IQC_discolor', 'IQC_mark', 'IQC_other', 'Others'
        ]

    def infer(self, image_list, *kwargs):
        """
        :image_list: list of images to query
        """
        distributions = self._classifier(image_list)
        results = list()
        for distribution in distributions:
            assert len(distribution) == len(self._classnames)
            results.append(
                dict(
                    scores = distribution,
                    classnames = self._classnames
                )
            )
        return results

    __call__ = infer
