import time
import sys
from threading import Thread
from numba import cuda
import tensorflow as tf

import glob

sys.path.append('../')
sys.path.append('../models/')
sys.path.append('../models/nn_models/')
sys.path.append('../models/nn_models/mrcnn/')
sys.path.append('../models/utils/')

from models.nn_models.MaskRCNN import myMaskRCNNConfig, MRCNNLogoInsertion
from models.nn_models.mrcnn import model as modellib
import cv2
from core.config import app
import os


def add_audio(params, fps):
    """
    Extract audio file from input video and add it to output video
    :param video_path: video path
    :return: output video name
    """
    video_name = params['saving_link'].split('/')[-1]
    print(params['saving_link'])
    saving_path = 'testing/result/'
    audio_name = f"audio_{video_name.split('.')[0]}.mp3"
    input_video = params['source_link']
    audio_path = 'testing/tmp_audio'
    output_audio = os.path.join(audio_path, audio_name)
    os.system(f'ffmpeg -i {input_video} {output_audio}')
    if os.path.exists(output_audio):
        output_video = saving_path + 'sound_' + video_name
        os.system(f'ffmpeg -i {params["saving_link"]} -i {output_audio} -codec copy -shortest {output_video}')
        os.remove(output_audio)
        os.remove(params["saving_link"])
        os.system(f"ffmpeg -i {output_video} -c:v libx265 -crf {fps} {params['saving_link']}")
        os.remove(output_video)
    else:
        to_save = os.path.join(saving_path, 'result_'+video_name)
        os.system(f"ffmpeg -i {params['saving_link']} -c:v libx265 -crf {fps} {to_save}")
        os.remove(params['saving_link'])


def process_video(config_path):

    logo_insertor = MRCNNLogoInsertion()
    logo_insertor.init_params(config_path)

    config = myMaskRCNNConfig()
    logo_insertor.model = modellib.MaskRCNN(mode="inference", config=config, model_dir='/')
    logo_insertor.model.load_weights(logo_insertor.config['model_weights_path'], by_name=True)
    source_link = logo_insertor.config['source_link']
    saving_link = logo_insertor.config['saving_link']

    print("Detection step")
    cap = cv2.VideoCapture(source_link)
    logo_insertor.fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            logo_insertor.detect_banner(frame)
        else:
            break

        if cap.get(1) % 1000 == 0:
            print(f"Still need to process {cap.get(cv2.CAP_PROP_FRAME_COUNT) - cap.get(1)} frames")

    cap.release()

    num_of_saved_masks = len(os.listdir(logo_insertor.config['mask_path']))

    print('Insertion step')

    logo_insertor.frame_num = 0
    logo_insertor.before_smoothing = False
    logo_insertor.init_params(config_path)

    cap = cv2.VideoCapture(source_link)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(saving_link, four_cc, logo_insertor.fps, (frame_width, frame_height), True)

    while cap.isOpened():
        ret, frame = cap.read()

        if cap.get(1) % 1000 == 0:
            print(f"Still need to insert {cap.get(cv2.CAP_PROP_FRAME_COUNT) - cap.get(1)} frames")

        if ret:
            logo_insertor.detect_banner(frame)
            logo_insertor.insert_logo()

            out.write(frame)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    out.release()

    add_audio(logo_insertor.config, logo_insertor.fps)

    files = glob.glob(app.config['MASK_PATH']+'/*.npy')
    for f in files:
        os.remove(f)

    return {"saved_mask": num_of_saved_masks,
            "detections": logo_insertor.num_detections,
            "insertions": logo_insertor.num_insertions}
