import os
import unittest

import numpy as np
import pandas as pd
import yaml
from execution import process_video

import cv2
import sys
sys.path.append('../')
sys.path.append('../models/')
sys.path.append('../models/nn_models/')
sys.path.append('../models/utils')

from smooth import smooth_points, line_equation, process_mask, smooth_series
from mask_processing import get_contour, create_background, found_corners

from mrcnn.config import Config
from MaskRCNN import myMaskRCNNConfig, MRCNNLogoInsertion
from mrcnn import model as modellib


class TestModel(unittest.TestCase):

    def setUp(self):
        self.params = 'test_config.yaml'

        with open(self.params, 'r') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        model_config = myMaskRCNNConfig()
        self.insertor = MRCNNLogoInsertion()
        self.insertor.model = modellib.MaskRCNN(mode="inference", config=model_config, model_dir='/')
        self.insertor.model.load_weights(self.config['model_weights_path'], by_name=True)

    def test_init_params(self):

        videos = ['test_1.mp4', 'test_2.mp4']
        video = videos[np.random.randint(2)]

        # source_link = '/usr/src/app/testing/videos/'
        # saving_link = '/usr/src/app/testing/result/'
        source_link = 'videos/'
        saving_link = 'result/'

        source = source_link + video
        saving = saving_link + video

        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)

        first_step = np.random.randint(2)

        if first_step:
            frame_nums = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps)
            shift = np.random.randint(10)
            start = np.random.randint(frame_nums)
            finish = start + shift + 1
            periods = {'period_1': {'start': start, 'finish': finish}}
            self.first_step = False
        else:
            periods = {}

        self.config['periods'] = periods
        self.config['source_link'] = source
        self.config['saving_link'] = saving

        test_dict = {}
        test_dict['periods'] = periods
        test_dict['saving_link'] = saving

        documents = 1

        with open(self.params, 'w') as write_file:
            documents = yaml.dump(self.config, write_file)

        self.assertEqual(test_dict['periods'], self.config['periods'])
        self.assertEqual(documents, None)

    def test_initialization(self):

        status = self.insertor.init_params(self.params)

        self.assertEqual(status, "The settings are set")

#    def test_detection(self):
#        source_video = self.config['source_link']
#        cap = cv2.VideoCapture(source_video)
#        self.insertor.fps = cap.get(cv2.CAP_PROP_FPS)
#        while cap.isOpened():
#            ret, frame = cap.read()
#            if ret:
#                self.insertor.detect_banner(frame)
#            else:
#                break
#
#        cap.release()

    def test_smoothing(self):

        self.assertEqual(self.insertor._MRCNNLogoInsertion__get_smoothed_points(), "Successful smoothing")

    def test_execution(self):

        self.assertEqual(process_video(self.params), True)


if __name__ == '__main__':
    unittest.main()


