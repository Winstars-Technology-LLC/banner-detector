import cv2 as cv
import numpy as np
from scipy import stats as st
from model_initialization import BannerReplacer
import yaml
from visa_parameters_setting import set_visa_parameters


class OpencvBannerInception(BannerReplacer):

    def __init__(self, template, frame, logo):
        self.template = template
        self.frame = frame
        self.logo = logo
        self.h_ravel = None
        self.diagonal_coordinates_list = []
        self.coordinates_list = []
        self.visa_p = {}

    def __detect_contour(self, matcher, min_match_count, dst_threshold, nfeatures, neighbours):
        self.frame = cv.imread(self.frame)
        gray_frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        self.template = cv.imread(self.template)
        gray_template = cv.cvtColor(self.template, cv.COLOR_BGR2GRAY)

        sift = cv.xfeatures2d.SIFT_create(nfeatures=nfeatures)

        kp1, des1 = sift.detectAndCompute(gray_template, None)
        kp2, des2 = sift.detectAndCompute(gray_frame, None)

        index_params = {'algorithm': matcher['index_params'][0], 'trees': matcher['index_params'][1]}
        search_params = {'checks': matcher['search_params']}
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=neighbours)

        good = []
        for m, n in matches:
            if m.distance < dst_threshold * n.distance:
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

    def __adjust_logo_color(self, cr_frame):
        frame_hsv = cv.cvtColor(cr_frame, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(frame_hsv)
        mean_s = np.mean(s).astype(int)
        self.logo = cv.imread(self.logo)
        logo_hsv = cv.cvtColor(self.logo, cv.COLOR_BGR2HSV)
        logo_h, logo_s, logo_v = cv.split(logo_hsv)
        mean_logo_s = np.mean(logo_s).astype(int)
        s_coeff = round(mean_s / mean_logo_s, 2)
        new_s_logo = (logo_s * s_coeff).astype('uint8')
        new_logo_hsv = cv.merge([logo_h, new_s_logo, logo_v])
        self.logo = cv.cvtColor(new_logo_hsv, cv.COLOR_HSV2BGR)
        return frame_hsv, h

    def __detect_banner_color(self, h, frame_hsv):
        self.h_ravel = h.ravel()
        self.h_ravel = self.h_ravel[self.h_ravel != 0]
        h_mode = st.mode(self.h_ravel)[0][0]
        h_low = round(h_mode * 0.5, 0).astype(int)
        h_high = round(h_mode * 1.5, 0).astype(int)

        low_color = np.array([h_low, 0, 0])
        high_color = np.array([h_high, 255, 255])
        color_mask = cv.inRange(frame_hsv, low_color, high_color)
        _, contours, _ = cv.findContours(color_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        return contours

    def __find_diagonal_contour_coordinates(self, contour):
        x_top_left = contour[:, 0, 0].min()
        x_bot_right = contour[:, 0, 0].max()
        y_top_left = contour[:, 0, 1].min()
        y_bot_right = contour[:, 0, 1].max()
        self.diagonal_coordinates_list = [x_top_left, x_bot_right, y_top_left, y_bot_right]
        return self.diagonal_coordinates_list

    def __find_contour_coordinates(self, cr_frame, contours):
        if len(contours) == 1:
            x_top_left, x_bot_right, y_top_left, y_bot_right = self.__find_diagonal_contour_coordinates(contours)

        else:
            coordinates_array = []
            for cnt in contours:
                x_top_left, x_bot_right, y_top_left, y_bot_right = self.__find_diagonal_contour_coordinates(cnt)
                coordinates_array.append([x_top_left, x_bot_right, y_top_left, y_bot_right])
            x_top_left = min([coordinates_array[i][0] for i in range(len(coordinates_array))])
            x_bot_right = max([coordinates_array[i][1] for i in range(len(coordinates_array))])
            y_top_left = min([coordinates_array[i][2] for i in range(len(coordinates_array))])
            y_bot_right = max([coordinates_array[i][3] for i in range(len(coordinates_array))])

        self.diagonal_coordinates_list = [x_top_left, x_bot_right, y_top_left, y_bot_right]
        l_int = x_top_left + (x_bot_right - x_top_left) * 0.1
        r_int = x_bot_right - (x_bot_right - x_top_left) * 0.1

        min_x_max_y = []
        max_x_min_y = []

        for cnt in contours:
            for x_y in cnt:
                if x_top_left <= x_y[0][0] <= l_int:
                    min_x_max_y.append(x_y[0][1])

                if r_int <= x_y[0][0] <= x_bot_right:
                    max_x_min_y.append(x_y[0][1])

        y_left_side = np.max(min_x_max_y)
        y_right_side = np.min(max_x_min_y)

        top_left = [x_top_left, y_top_left]
        bot_left = [x_top_left, y_left_side]
        bot_right = [x_bot_right, y_bot_right]
        top_right = [x_bot_right, y_right_side]
        self.coordinates_list = [top_left, bot_left, bot_right, top_right]

        banner_mask_cr = cr_frame.copy()
        pts = np.array([top_left, bot_left, bot_right, top_right], np.int32)
        cv.fillPoly(banner_mask_cr, [pts], (0, 0, 255))
        return banner_mask_cr, y_left_side

    def __adjust_referee_colors(self, hsv_referee, area_threshold, frame_hsv, banner_mask_cr, coef, area_thre):
        low_ref = np.array([hsv_referee['low_h'], 0, hsv_referee['low_v']])
        high_ref = np.array([hsv_referee['high_h'], 255, hsv_referee['high_v']])
        referee_mask = cv.inRange(frame_hsv, low_ref, high_ref)

        _, contours, _ = cv.findContours(referee_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > area_threshold:
                cv.drawContours(banner_mask_cr, [cnt], 0, (0, 255, 0), -1)

                # body parts
                al = cnt[:, 0, 0].min()  # X coordinate for top left corner
                bl = cnt[:, 0, 0].max()  # X coordinate for bottom right corner
                cl = cnt[:, 0, 1].min()  # Y coordinate for top left corner
                dl = cnt[:, 0, 1].max()  # Y coordinate for bottom right corner

                ref_cr_hsv = frame_hsv[int(cl * coef['1']):int(dl * coef['2']), int(al * coef['3']):int(bl * coef['4'])]
                banner_mask_cr_ref = banner_mask_cr[int(cl * coef['1']):int(dl * coef['2']),
                                     int(al * coef['3']):int(bl * coef['4'])]

                # body color
                low_bd = np.array([100, 10, 150])
                high_bd = np.array([200, 70, 255])
                body_mask = cv.inRange(ref_cr_hsv, low_bd, high_bd)

                _, contours, _ = cv.findContours(body_mask, cv.RETR_TREE,
                                                 cv.CHAIN_APPROX_SIMPLE)
                # drawing contour
                for cnt2 in contours:
                    area = cv.contourArea(cnt2)
                    if area > area_thre:
                        cv.drawContours(banner_mask_cr_ref, [cnt2], 0, (0, 255, 0), -1)

        # flag color
        low_flg = np.array([0, 50, 150])
        high_flg = np.array([50, 200, 255])
        flag_mask = cv.inRange(frame_hsv, low_flg, high_flg)

        _, contours, _ = cv.findContours(flag_mask, cv.RETR_TREE,
                                         cv.CHAIN_APPROX_SIMPLE)
        # drawing contour
        for cnt3 in contours:
            area = cv.contourArea(cnt3)
            if area > 1:
                cv.drawContours(banner_mask_cr, [cnt3], 0, (0, 255, 0), -1)

    def __resize_banner(self, y_left_side, resize_coef, w_threshold):
        top_left, bot_left, bot_right, top_right = self.coordinates_list
        x_top_left, x_bot_right, y_top_left, y_bot_right = self.diagonal_coordinates_list
        w = x_bot_right - x_top_left  # detected area width after resizing
        h = y_bot_right - y_top_left  # detected area height after resizing
        banner_height = y_left_side - y_top_left
        banner_width = w
        pred_width = int(banner_height * resize_coef)  # predicted width using proportions

        if w < w_threshold * pred_width:
            banner_width = pred_width
        xres = cv.resize(self.logo, (banner_width, h))  # resized banner

        rtx = [x_bot_right, y_bot_right - h]  # top right corner coordinates before transformation
        lbx = [x_bot_right - w, y_bot_right]  # left bottom corner coordinates before transformation

        pts1 = np.float32([[top_left, rtx, lbx, bot_right]])
        pts2 = np.float32([[top_left, top_right, bot_left, bot_right]])
        mtrx = cv.getPerspectiveTransform(pts1, pts2)
        resized_banner = cv.warpPerspective(xres, mtrx, (w, h), borderMode=1)
        return h, w, resized_banner

    def __insert_ads(self, banner_mask_cr, cr_frame, resized_banner, h, w, frame):
        for i in range(self.coordinates_list[0][1], h):
            for j in range(self.coordinates_list[0][0], w):
                if list(banner_mask_cr[i, j]) == [0, 0, 255]:
                    if list(banner_mask_cr[i, j]) == [0, 255, 0]:
                        continue
                    cr_frame[i, j] = resized_banner[i, j]
        cv.imshow('Replaced', frame)
        cv.waitKey(0)

    def build_model(self):
        with open('visa_parameters.yml', 'r') as stream:
            self.visa_p = yaml.safe_load(stream)

    def detect_banner(self):
        switch, cr_frame = self.__detect_contour(self.visa_p['matcher'], self.visa_p['min_match_count'],
                                                 self.visa_p['dst_threshold'], self.visa_p['nfeatures'],
                                                 self.visa_p['neighbours'])
        frame_hsv, h = self.__adjust_logo_color(cr_frame)
        contours = self.__detect_banner_color(h, frame_hsv)
        banner_mask_cr, y_left_side = self.__find_contour_coordinates(cr_frame, contours)
        self.__adjust_referee_colors(self.visa_p['hsv_referee'], self.visa_p['area_threshold'], frame_hsv,
                                     banner_mask_cr, self.visa_p['coef'], self.visa_p['area_thre'])
        h, w, resized_banner = self.__resize_banner(y_left_side, self.visa_p['resize_coef'], self.visa_p['w_threshold'])
        return h, w, banner_mask_cr, cr_frame, resized_banner

    def insert_banner(self):
        h, w, banner_mask_cr, cr_frame, resized_banner = self.detect_banner()
        self.__insert_ads(banner_mask_cr, cr_frame, resized_banner, h, w, self.frame)


if __name__ == '__main__':
    # set_visa_parameters()
    opencv_inception = OpencvBannerInception('SET TEMPLATE', 'SET FRAME', 'SET BANNER')
    opencv_inception.build_model()
    opencv_inception.insert_banner()
