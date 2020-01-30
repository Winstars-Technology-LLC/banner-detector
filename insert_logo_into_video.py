import cv2 as cv
from OpencvLogoInsertion import OpencvLogoInsertion


def get_additional_templates():
    """ Get additional templates for detection """
    add_tmp1 = 'SET ADDITIONAL TEMPLATE NAME'
    add_tmp2 = 'SET ADDITIONAL TEMPLATE NAME'
    return [add_tmp1, add_tmp2]


def insert_logo_into_video(video, write_video=True):
    """
    This function provides logo insertion into the video file

    :param video: input video file
    :param write_video: choose True if you want to write the video
    :return:
    """
    add_templates = get_additional_templates()
    capture = cv.VideoCapture(video)
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    four_cc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter('RESULTING VIDEO NAME', four_cc, 30, (frame_width, frame_height), True)

    while capture.isOpened():
        ret, frame = capture.read()
        if ret:
            opencv_insertion = OpencvLogoInsertion('SET TEMPLATE', frame, 'SET LOGO')
            opencv_insertion.build_model('SET PARAMETERS')
            cropped_frame, resized_banner_, banner_mask_cr_, switch_ = opencv_insertion.detect_banner()
            opencv_insertion.insert_logo(cropped_frame, resized_banner_, banner_mask_cr_, switch_)

            if not switch_:
                for tmp in add_templates:
                    opencv_insertion = OpencvLogoInsertion(tmp, frame, 'SET LOGO')
                    opencv_insertion.build_model('SET PARAMETERS')
                    cropped_frame, resized_banner_, banner_mask_cr_, switch_ = opencv_insertion.detect_banner()
                    opencv_insertion.insert_logo(cropped_frame, resized_banner_, banner_mask_cr_, switch_)
                    if switch_:
                        break
                    else:
                        continue

            if write_video:
                out.write(frame)

            key = cv.waitKey(1)
            if key == 27:
                break
        else:
            break

    capture.release()
    out.release()


insert_logo_into_video('SET INPUT VIDEO NAME')
