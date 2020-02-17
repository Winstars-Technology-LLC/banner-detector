import cv2 as cv
import numpy as np
from OpenCVLogoInsertion import OpenCVLogoInsertion
from banner_parameters_setting import banner_parameters_setting


def get_additional_templates():
    """
    Get additional templates for better banner detection

    :return: list of additional templates
    """
    add_tmp1 = 'SET ADDITIONAL TEMPLATE'
    add_tmp2 = 'SET ADDITIONAL TEMPLATE'
    return [add_tmp1, add_tmp2]


def frames_capture(video, show_details=False):
    capture = cv.VideoCapture(video)
    frames_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv.CAP_PROP_FPS)

    if show_details:
        print('Total amount of frames:', frames_count)
        print('FPS amount:', fps)

    frames_list = []

    while capture.isOpened():
        res, image = capture.read()
        if res:
            frames_list.append(image)
        else:
            capture.release()
    for i in range(0, frames_count):
        cv.imwrite('SET FOLDER PATH WHERE TO PASTE FRAMES/frame{0}.jpg'.format(i), frames_list[i])


def insert_logo_into_video(video, write_video=True, show_f1=False):
    """
    This function provides logo insertion into the video file

    :param video: input video file
    :param write_video: choose True if you want to write the video
    :param show_f1: True if need to calculate f1 score
    :return: mean value for f1 score
    """
    add_templates = get_additional_templates()
    # banner_parameters_setting()

    capture = cv.VideoCapture(video)

    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    four_cc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter('SET RESULTING VIDEO NAME', four_cc, 30, (frame_width, frame_height), True)

    n = 0
    f1_list = []
    while capture.isOpened():
        ret, frame = capture.read()
        if ret:
            frame_n = 'frame{}.jpg'.format(n)

            open_cv_insertion = OpenCVLogoInsertion('SET TEMPLATE NAME', frame, 'SET LOGO NAME')
            open_cv_insertion.build_model('SET PARAMETERS')
            cropped_frame, resized_banner, banner_mask_cr_, switch_, f1 = open_cv_insertion.detect_banner(frame_n)
            open_cv_insertion.insert_logo(cropped_frame, resized_banner, banner_mask_cr_, switch_)

            if not switch_:
                for tmp in add_templates:
                    open_cv_insertion = OpenCVLogoInsertion(tmp, frame, 'SET LOGO NAME')
                    open_cv_insertion.build_model('SET PARAMETERS')
                    cropped_frame, resized_banner, banner_mask_cr_, switch_, f1\
                        = open_cv_insertion.detect_banner(frame_n)
                    open_cv_insertion.insert_logo(cropped_frame, resized_banner, banner_mask_cr_, switch_)
                    if switch_:
                        break
                    else:
                        continue

            if f1 != -1 and show_f1:
                f1_list.append(f1)

            if write_video:
                out.write(frame)

            key = cv.waitKey(1)
            if key == 27:
                break
            n += 1
        else:
            break
    capture.release()
    out.release()
    if len(f1_list) > 0:
        return np.mean(f1_list)
    else:
        pass


# frames_capture('SET INPUT VIDEO NAME')
performance_evaluation = insert_logo_into_video('SET INPUT VIDEO NAME')
# print(performance_evaluation)
