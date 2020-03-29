import cv2
import yaml
import pandas as pd

from BannerReplacer import BannerReplacer


class Template(BannerReplacer):

    def __init__(self):
        self.model = None
        self.model_params = None
        self.config = None
        self.frame = None
        self.detection_successful = False
        self.fps = None
        self.num_frames = None
        self.start = None
        self.finish = None
        self.period = None
        self.process = False
        self.key = None
        self.frame_num = 0
        self.saved_points = pd.DataFrame(columns=['x_top_left', 'y_top_left', 'x_top_right',
                                                  'y_top_right', 'x_bot_left', 'y_bot_left',
                                                  'x_bot_right', 'y_bot_right'])


    def init_params(self, params):
        """
        reading parameters in python dictionary
        :param params:
        :return:
        """
        with open(params) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.key = self.config['periods'].keys()[0]
        self.period = self.config['periods'][self.key]
        self.start, self.finish = self.period.values()

    def detect_banner(self):

        self.__valid_time()

        if self.process:
            # detect banner
        else:
            pass

    def build_model(self, params):
        pass

    def __valid_time(self):
        """
        checks time intervals
        :return:
        """
        # self.frame_num = self.fps * self.start - 1
        time = self.frame_num / self.fps
        if self.start <= time and time <= self.finish:
            self.process = True
        else:
            self.process = False

        if time == self.finish:
            self.saved_points.to_csv(self.key+'.csv')
            del self.config['periods'][self.key]
            self.key = self.config['periods'].keys()[0]
            self.period = self.config['periods'][self.key]
            self.start, self.finish = self.period.values()



if __name__ == '__main__':
    logo_insertor = Template()
    logo_insertor.init_params('template.yaml')

    source_video = logo_insertor.config['video_path']
    result_video_path = logo_insertor.config['result_video_path']

    cap = cv2.VideoCapture(source_video)
    ret = 1
    while ret:
        ret, frame = cap.read()

        if ret:
            logo_insertor.detect_banner()
    cap.release()

    logo_insertor.frame_num = 0
    logo_insertor.smooth = False

    cap = cv2.VideoCapture(source_video)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(result_video_path, four_cc, logo_insertor.fps, (frame_width, frame_height), True)

    ret = 1
    while ret:

        ret, frame = cap.read()

        if ret:
            logo_insertor.detect_banner()
            logo_insertor.insert_logo()

            out.write(frame)

            cv2.imshow('Video (press Q to close)', frame)
            if cv2.waitKey(23) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        out.release()
