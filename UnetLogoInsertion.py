import tensorflow as tf
import os
import numpy as np
import cv2
import math
import yaml
from BannerReplacer import BannerReplacer


class UnetLogoInsertion(BannerReplacer):
    '''
    The model detects banner and replace it with other logo using Unet neural network model
    '''

    def __init__(self):
        self.model = None
        self.detected_mask = None
        self.detection_successful = False
        self.frame = None
        self.old_frame_gray = None
        self.old_points = None
        self.model_parameters = None
        self.corners = None
        self.first_frame = True
        self.old_width = None
        self.old_width_2 = None

    def build_model(self, parameters_filepath):
        '''
        This method builds Unet neural network model and load trained weights
        :parameters_filepath: load model parameters from YAML file
        '''
        # loading and saving model parameters to class attribute
        with open(parameters_filepath, 'r') as file:
            self.model_parameters = yaml.safe_load(file)

        # load parameters
        img_height = self.model_parameters['img_height']
        img_width = self.model_parameters['img_width']
        img_channels = self.model_parameters['img_channels']
        model_weights_path = self.model_parameters['model_weights_path']
        train_model = self.model_parameters['train_model']

        inputs = tf.keras.layers.Input((img_height, img_width, img_channels))

        # CONTRACTION PATH
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(
            inputs)
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        p1 = tf.keras.layers.Dropout(0.2)(p1)

        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
        p2 = tf.keras.layers.Dropout(0.2)(p2)

        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
        p3 = tf.keras.layers.Dropout(0.2)(p3)

        c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
        p4 = tf.keras.layers.Dropout(0.2)(p4)

        c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        # EXPANSIVE PATH
        u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        u6 = tf.keras.layers.Dropout(0.2)(u6)
        c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        u7 = tf.keras.layers.Dropout(0.2)(u7)
        c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        u8 = tf.keras.layers.Dropout(0.2)(u8)
        c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        u9 = tf.keras.layers.Dropout(0.2)(u9)
        c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
        self.model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        self.model.compile(optimizer='adam', loss=self.__loss, metrics=[self.__dice_coef])

        # training model if required
        if train_model:
            x_train_path = self.model_parameters['x_train_path']
            y_train_path = self.model_parameters['y_train_path']
            self.__train_model(x_train_path, y_train_path, img_height, img_width, img_channels, model_weights_path)

        # load trained model weights
        self.model.load_weights(model_weights_path)

    def detect_banner(self, frame):
        '''
        This method detects banner's pixels using Unet model, and saves deteсted binary mask
        and saves coordinates for top left and bottom right corners of a banner
        :frame: image or video frame where we will make detection and insertion
        '''
        self.frame = frame

        # load parameters
        value_threshold = self.model_parameters['value_threshold']
        filter_area_size = self.model_parameters['filter_area_size']

        # getting full size predicted mask of the frame
        fsz_mask = self.__predict_full_size()
        fsz_mask = (fsz_mask > value_threshold).astype(np.uint8)

        # looking for contours
        first_cnt = True
        _, thresh = cv2.threshold(fsz_mask, value_threshold, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > filter_area_size:

                # works for the first contour
                if first_cnt:
                    x_top_left = cnt[:, 0, 0].min()
                    x_bot_right = cnt[:, 0, 0].max()
                    y_top_left = cnt[:, 0, 1].min()
                    y_bot_right = cnt[:, 0, 1].max()
                    first_cnt = False

                # works with more than one contour, and replace coordinates with more relevant
                else:
                    new_x_top_left = cnt[:, 0, 0].min()
                    if new_x_top_left < x_top_left:
                        x_top_left = new_x_top_left

                    new_x_bot_right = cnt[:, 0, 0].max()
                    if new_x_bot_right > x_bot_right:
                        x_bot_right = new_x_bot_right

                    new_y_top_left = cnt[:, 0, 1].min()
                    if new_y_top_left < y_top_left:
                        y_top_left = new_y_top_left

                    new_y_bot_right = cnt[:, 0, 1].max()
                    if new_y_bot_right > y_bot_right:
                        y_bot_right = new_y_bot_right

                cv2.drawContours(fsz_mask, [cnt], -1, (1), -1)

        # save detected mask as a class attribute
        self.detected_mask = fsz_mask

        if first_cnt:
            return

        # save corners coordinates to class attribute
        self.corners = [(x_top_left, y_top_left), (x_bot_right, y_bot_right)]

        # improving coordinate using optical flow
        self.__check_optical_flow()
        self.old_points = np.array([self.corners[0]], dtype=np.float32)

        # set that the banner detection was successful
        self.detection_successful = True

    def insert_logo(self):
        '''
        This method insert logo into detected area on the frame
        '''
        if not self.detection_successful:
            return

        # load parameters
        logo = cv2.imread(self.model_parameters['logo_link'], cv2.IMREAD_UNCHANGED)
        height_coef = self.model_parameters['height_coef']
        width_coef = self.model_parameters['width_coef']

        x_top_left = self.corners[0][0]
        y_top_left = self.corners[0][1]
        x_bot_right = self.corners[1][0]
        y_bot_right = self.corners[1][1]

        # banner height before transformation
        rect_height = y_bot_right - y_top_left

        # banner side height after transformation
        height = rect_height * height_coef

        # banner width
        width = (int(math.ceil((rect_height * width_coef) / 10.0)) * 10)

        # keep same width if the changes was on only one frame
        width = self.__adjust_logo_width(width)

        # adjust logo to banner's shape
        transformed_logo = self.__adjust_logo_shape(logo, rect_height, height, width)

        # check the end of the banner on the frame
        check_end_frame = lambda start, end, length: start + end if (start + end <= length) else length
        end_y = check_end_frame(y_top_left, rect_height, self.frame.shape[0])
        end_x = check_end_frame(x_top_left, width, self.frame.shape[1])

        # adjust logo color to the banner area
        frame_cr = self.frame[y_top_left:end_y, x_top_left:end_x].copy()
        transformed_logo = self.__logo_color_adj(transformed_logo, frame_cr)

        # replacing banner pixels with logo pixels
        for k in range(y_top_left, end_y):
            for j in range(x_top_left, end_x):
                if self.detected_mask[k, j] == 1:
                    self.frame[k, j] = transformed_logo[k - y_top_left, j - x_top_left]

    def __train_model(self, x_train_path, y_train_path, img_height, img_width, img_channels, model_weights_path):
        '''
        This method trains new model using X and Y train datasets
        :x_train_path: the path to X train dataset
        :y_train_path: the path to Y train dataset in .npy format
        :img_height: train image height
        :img_width: train image width
        :img_channels: number of channels for train image
        :model_weights_path: model weight path for saving
        '''
        # looking for files
        train_x_list = next(os.walk(x_train_path))[2]

        # empty lists for X_traind and Y_train
        x_train = np.zeros((len(train_x_list), img_height, img_width, img_channels), dtype=np.float32)
        y_train = np.zeros((len(train_x_list), img_height, img_width, 1), dtype=np.float32)

        # replacing empty lists elements with actual train data
        n = 0
        for file in train_x_list:
            x_train_file = x_train_path + file
            id_, _ = file.split('.')
            y_train_file = y_train_path + id_ + '.npy'
            x = cv2.imread(x_train_file, cv2.IMREAD_UNCHANGED)
            y = np.load(y_train_file)
            y_ = np.expand_dims(y, axis=-1)
            x_train[n] = x
            y_train[n] = y_
            n += 1

        # setting callbacks for the model
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0),
                     tf.keras.callbacks.ModelCheckpoint(model_weights_path, monitor='val_loss', verbose=0,
                                                        save_best_only=True, save_weights_only=True)]

        # training the model
        self.model.fit(x_train, y_train, validation_split=0.1, epochs=200, callbacks=callbacks)

    def __loss(self, y_true, y_pred):
        '''
        Creating combined BCE and Dice Loss function which we will use in the model
        :y_true: actual Y values of test data
        :y_pred: predicted Y values of test data
        :return: combined BCE and Dice Loss function
        '''
        return tf.keras.losses.binary_crossentropy(y_true, y_pred) + self.__dice_loss(y_true, y_pred)

    def __dice_loss(self, y_true, y_pred):
        '''
        Creating a Dice Loss function for our model
        :y_true: actual Y values of test data
        :y_pred: predicted Y values of test data
        :return: Dice Loss function
        '''
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
        return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

    def __dice_coef(self, y_true, y_pred):
        '''
        Creating a Dice Coefficient to use it like a metric in our model
        :y_true: actual Y values of test data
        :y_pred: predicted Y values of test data
        :return: Dice Coefficient
        '''
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
        denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
        return (numerator + 1) / (denominator + 1)

    def __check_optical_flow(self):
        '''
        Works for video, predicts where the previous point is supposed to be
        on the next frame
        :return: the point that was predicted and the GRAY frame which need to be used for the next frame
        '''
        gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        # load parameters
        diff = self.model_parameters['diff']
        winSize = self.model_parameters['winSize']
        maxLevel = self.model_parameters['maxLevel']

        lk_params = dict(winSize=(winSize, winSize),
                         maxLevel=maxLevel,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        if self.first_frame:
            self.old_frame_gray = gray_frame.copy()
            self.first_frame = False
            return

        new_points_flow, status, error = cv2.calcOpticalFlowPyrLK(self.old_frame_gray, gray_frame, self.old_points,
                                                                  None, **lk_params)
        new_x, new_y = new_points_flow.ravel()

        self.old_frame_gray = gray_frame.copy()
        if abs(self.corners[0][0] - new_x) < diff and abs(self.corners[0][1] - new_y) < diff:
            self.corners[0] = int(round(new_x)), int(round(new_y))

    def __adjust_logo_width(self, width):
        '''
        The method dellays width changing, and if the change was only for one frame
        and then returned back, it keeps same width without changing
        :return: adjusted width
        '''
        temp_width = width

        if width != self.old_width and width != self.old_width_2 and self.old_width != None:
            temp_width = self.old_width

        self.old_width_2 = self.old_width
        self.old_width = width
        width = temp_width
        return width

    def __adjust_logo_shape(self, logo, rect_height, height, width):
        '''
        The method resizes and applies perspective transformation on logo
        :logo: the logo that we will transform
        :rect_height: height of the rectangular logo before transformation
        :height: height of the logo after transformation
        :width: width of the logo
        :return: transformed logo
        '''
        # resize the logo
        resized_logo = cv2.resize(logo, (width, rect_height))

        # transform the logo
        pts1 = np.float32([(0, 0), (0, rect_height), (width, rect_height), (width, 0)])
        pts2 = np.float32([(0, 0), (0, height), (width, rect_height), (width, rect_height - height)])
        mtrx = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_logo = cv2.warpPerspective(resized_logo, mtrx, (width, rect_height), borderMode=1)
        return transformed_logo

    def __logo_color_adj(self, logo, banner):
        '''
        The method changes color of the logo to adjust it to frame
        :logo: the logo that we will change
        :banner: area of detected banner
        :return: changed logo
        '''
        # get logo hsv
        logo_hsv = cv2.cvtColor(logo, cv2.COLOR_BGR2HSV)
        logo_h, logo_s, logo_v = cv2.split(logo_hsv)

        # get banner hsv
        banner_hsv = cv2.cvtColor(banner, cv2.COLOR_BGR2HSV)
        _, banner_s, _ = cv2.split(banner_hsv)

        # find the saturation difference between both images
        mean_logo_s = np.mean(logo_s).astype(int)
        mean_banner_s = np.mean(banner_s).astype(int)
        trans_coef = round(mean_banner_s / mean_logo_s, 2)

        # adjust logo saturation according to the difference
        adjusted_logo_s = (logo_s * trans_coef).astype('uint8')
        adjusted_logo_hsv = cv2.merge([logo_h, adjusted_logo_s, logo_v])
        adjusted_logo = cv2.cvtColor(adjusted_logo_hsv, cv2.COLOR_HSV2BGR)

        return adjusted_logo

    def __predict_full_size(self):
        '''
        The method goes trougth the frame and detects smaller areas,
        then combines them togeather
        :return: full size mask with detected banner pixels
        '''
        # load parameters
        img_height = self.model_parameters['img_height']
        img_width = self.model_parameters['img_width']
        step = self.model_parameters['full_size_step']

        # getting the frame size
        frame_height, frame_width, _ = self.frame.shape

        # create mask for full size image prediction
        fsz_mask = np.zeros((frame_height, frame_width, 1), dtype='float32')

        flag_k = False
        flag_j = False

        # split up the full frame to smaller images (same than using in model) and predict them
        for k in range(0, frame_height, step):
            if k + img_height >= (frame_height - 1):
                k = frame_height - img_height
                flag_k = True

            for j in range(0, frame_width, step):
                if j + img_width >= (frame_width - 1):
                    j = frame_width - img_width
                    flag_j = True

                mask_cr = fsz_mask[k:k + img_height, j:j + img_width]
                frame_cr = self.frame[k:k + img_height, j:j + img_width]
                test_cr = np.expand_dims(frame_cr, axis=0)
                cr_predict = self.model.predict(test_cr)

                for y in range(img_height):
                    for x in range(img_width):
                        if cr_predict[0][y, x] > mask_cr[y, x]:
                            mask_cr[y, x] = cr_predict[0][y, x]

                if flag_j:
                    break

            if flag_k:
                break

        return fsz_mask


if __name__ == '__main__':

    logo_insertor = UnetLogoInsertion()
    logo_insertor.build_model('model_parameters_setting')

    # load parameters
    source_type = logo_insertor.model_parameters['source_type']
    source_link = logo_insertor.model_parameters['source_link']
    save_result = logo_insertor.model_parameters['save_result']
    saving_link = logo_insertor.model_parameters['saving_link']

    # works with video
    if source_type == 0:
        cap = cv2.VideoCapture(source_link)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        four_cc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(saving_link, four_cc, 30, (frame_width, frame_height), True)

        while (cap.isOpened()):
            ret, frame = cap.read()

            if ret:
                logo_insertor.detect_banner(frame)
                logo_insertor.insert_logo()

                if save_result:
                    out.write(frame)

                cv2.imshow('Video (press Q to close)', frame)
                if cv2.waitKey(23) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
        out.release()

    # works with image
    else:
        frame = cv2.imread(source_link, cv2.IMREAD_UNCHANGED)
        logo_insertor.detect_banner(frame)
        logo_insertor.insert_logo()
        if save_result:
            cv2.imwrite(saving_link, frame)
        cv2.imshow('Image (press Q to close)', frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
