import cv2 as cv
import numpy as np
from scipy import stats as st
from abc import ABC, abstractmethod


class BannerInception(ABC):
    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def banner_detection(self):
        pass

    @abstractmethod
    def banner_insertion(self):
        pass


class OpencvBannerInception(BannerInception):

    def __init__(self, template, frame, logo):
        self.template = template
        self.frame = frame
        self.logo = logo

    def __contour_detection(self, matcher_index_params, matcher_search_params, min_match_count, dst_coef):
        self.frame = cv.imread(self.frame)
        gray_frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        self.template = cv.imread(self.template)
        gray_template = cv.cvtColor(self.template, cv.COLOR_BGR2GRAY)

        sift = cv.xfeatures2d.SIFT_create(nfeatures=200000)

        kp1, des1 = sift.detectAndCompute(gray_template, None)
        kp2, des2 = sift.detectAndCompute(gray_frame, None)

        flann = cv.FlannBasedMatcher(matcher_index_params, matcher_search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < dst_coef * n.distance:
                good.append(m)
        cr_frame = None
        if len(good) >= min_match_count:
            switch = True
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
            m, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            h, w = gray_template.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, m)

            x_corner_list = [dst[i][0][0] for i in range(len(dst))]
            y_corner_list = [dst[j][0][1] for j in range(len(dst))]
            x_min, x_max = np.int64(min(x_corner_list)), np.int64(max(x_corner_list))
            y_min, y_max = np.int64(min(y_corner_list)), np.int64(max(y_corner_list))
            cr_frame = self.frame[y_min:y_max, x_min:x_max]
        else:
            switch = False
        return switch, cr_frame

    def __logo_color_adjustment(self, cr_frame):
        frame_hsv = cv.cvtColor(cr_frame, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(frame_hsv)
        mean_s = np.mean(s).astype(int)
        logo_hsv = cv.cvtColor(self.logo, cv.COLOR_BGR2HSV)
        logo_h, logo_s, logo_v = cv.split(logo_hsv)
        mean_logo_s = np.mean(logo_s).astype(int)
        s_coeff = round(mean_s / mean_logo_s, 2)
        new_s_logo = (logo_s * s_coeff).astype('uint8')
        new_logo_hsv = cv.merge([logo_h, new_s_logo, logo_v])
        self.logo = cv.cvtColor(new_logo_hsv, cv.COLOR_HSV2BGR)

    def __banner_color_detection(self, h, frame_hsv):
        h_ravel = h.ravel()
        h_ravel = h_ravel[h_ravel != 0]
        h_mode = st.mode(h_ravel)[0][0]
        h_low = round(h_mode * 0.5, 0).astype(int)
        h_high = round(h_mode * 1.5, 0).astype(int)

        low_color = np.array([h_low, 0, 0])
        high_color = np.array([h_high, 255, 255])
        color_mask = cv.inRange(frame_hsv, low_color, high_color)
        _, self.contours, _ = cv.findContours(color_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        return self.contours

    def __find_diagonal_contour_coordinates(self, contour):
        self.x_top_left = contour[:, 0, 0].min()
        self.x_bot_right = contour[:, 0, 0].max()
        self.y_top_left = contour[:, 0, 1].min()
        self.y_bot_right = contour[:, 0, 1].max()
        return self.x_top_left, self.x_bot_right, self.y_top_left, self.y_bot_right

    def __find_contour_coordinates(self, cr_frame):
        if len(self.contours) == 1:
            x_top_left, x_bot_right, y_top_left, y_bot_right = self.__find_diagonal_contour_coordinates(self.contours)

        else:
            coordinates_array = []
            for cnt in self.contours:
                x_top_left, x_bot_right, y_top_left, y_bot_right = self.__find_diagonal_contour_coordinates(cnt)
                coordinates_array.append([x_top_left, x_bot_right, y_top_left, y_bot_right])
            x_top_left = min([coordinates_array[i][0] for i in range(len(coordinates_array))])
            x_bot_right = max([coordinates_array[i][1] for i in range(len(coordinates_array))])
            y_top_left = min([coordinates_array[i][2] for i in range(len(coordinates_array))])
            y_bot_right = max([coordinates_array[i][3] for i in range(len(coordinates_array))])

        l_int = x_top_left + (x_bot_right - x_top_left) * 0.1
        r_int = x_bot_right - (x_bot_right - x_top_left) * 0.1

        min_x_max_y = []
        max_x_min_y = []

        for x_y in self.contours:
            if x_top_left <= x_y[0][0] <= l_int:
                min_x_max_y.append(x_y[0][1])

            if r_int <= x_y[0][0] <= x_bot_right:
                max_x_min_y.append(x_y[0][1])

        y_left_side = np.max(min_x_max_y)
        y_right_side = np.min(max_x_min_y)

        top_left = [x_top_left, y_top_left]  # top left
        bot_left = [x_top_left, y_left_side]  # bottom left
        bot_right = [x_bot_right, y_bot_right]  # bottom right
        top_right = [x_bot_right, y_right_side]  # top right

        banner_mask_cr = cr_frame.copy()
        pts = np.array([top_left, bot_left, bot_right, top_right], np.int32)
        cv.fillPoly(banner_mask_cr, [pts], (0, 0, 255))

