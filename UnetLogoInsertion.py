from BannerReplacer import BannerReplacer
import tensorflow as tf
import numpy as np
import cv2


class UnetLogoInsertion(BannerReplacer):
    '''
    The model detects banner and replace it with other logo using Unet neural network model
    '''

    def __init__(self, frame, logo):
        self.model = None
        self.detected_mask = None
        self.frame = frame
        self.logo = logo

    def build_model(self, model_weights_path, img_height, img_width, img_channels):

        '''
        This method builds Unet neural network model and load trained weights
        :model_weights_path: the path to the saved weights for the model
        :img_width: width of the input image for training
        :img_height: height of the input image for training
        :img_channels: number of channels in input image
        '''

        inputs = tf.keras.layers.Input((img_width, img_height, img_channels))

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

        # load trained model weights
        self.model.load_weights(model_weights_path)

    def detect_banner(self):

        # getting full size predicted mask of the frame
        fsz_mask = self.__predict_full_size(img_height=256, img_width=256)

        fsz_mask = (fsz_mask > 0.95).astype(np.uint8)

        # looking for contours
        first_cnt = True
        min_x_max_y = []
        max_x_min_y = []

        _, thresh = cv2.threshold(fsz_mask, 0.5, 255, 0)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 30:

                # works for the first contour
                if first_cnt:
                    x_top_left = cnt[:, 0, 0].min()
                    x_bot_right = cnt[:, 0, 0].max()
                    y_top_left = cnt[:, 0, 1].min()
                    y_bot_right = cnt[:, 0, 1].max()

                    first_cnt = False

                # works with more than one contour
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

    def insert_logo(self):
        pass

    def __loss(self, y_true, y_pred):
        '''
        Creating combined BCE and Dice Loss function which we will use in the model
        '''
        return tf.keras.losses.binary_crossentropy(y_true, y_pred) + self.__dice_loss(y_true, y_pred)

    def __dice_loss(self, y_true, y_pred):
        '''
        Creating a Dice Loss function for our model
        '''

        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
        return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

    def __dice_coef(self, y_true, y_pred):
        '''
        Creating a Dice Coefficient to use it like a metric in our model
        '''
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
        denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
        return (numerator + 1) / (denominator + 1)

    def __optical_flow_point():
        pass

    def __predict_full_size(self, img_height, img_width, step=50):

        # getting the frame size
        frame_height, frame_width, _ = self.frame.shape

        # create mask for full size i,ahe prediction
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
                frame_cr = frame[k:k + img_height, j:j + img_width]
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
