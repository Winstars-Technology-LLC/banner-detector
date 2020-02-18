import cv2 as cv
import numpy as np
from scipy import stats as st
from BannerReplacer import BannerReplacer
import yaml
from banner_parameters_setting import banner_parameters_parameters


class OpencvLogoInsertion(BannerReplacer):
    """
    The model provides logo insertion with the OpenCV package
    """
    def __init__(self, template, frame, logo):
        self.template = template
        self.frame = frame
        self.logo = logo
        self.contours = []
        self.diagonal_coordinates_list = []
        self.coordinates_list = []
        self.template_p = {}

    def __detect_contour(self, matcher, min_match_count, dst_threshold, nfeatures, neighbours, rc_threshold):
        """
        The method provides the detection of the field where the logo must be inserted

        :param matcher:  tunes the Matcher object
        :param min_match_count:  the minimum quantity of matched keypoints between frame and logo to detect the field
        :param dst_threshold: the threshold for the distance between matched descriptors
        :param nfeatures: the number of features for the SIFT algorithm
        :param neighbours: the amount of best matches found per each query descriptor
        :param rc_threshold: the threshold for the Homographies mask
        :return: switch that indicates whether the required field was found or not; the required field, the required
        field in hsv mode
        """
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
        frame_hsv = None
        if len(good) >= min_match_count:
            switch = True
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
            m, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, rc_threshold)
            h, w = gray_template.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, m)
            x_corner_list = [dst[i][0][0] for i in range(len(dst))]
            y_corner_list = [dst[j][0][1] for j in range(len(dst))]
            x_min, x_max = np.int64(min(x_corner_list)), np.int64(max(x_corner_list))
            y_min, y_max = np.int64(min(y_corner_list)), np.int64(max(y_corner_list))
            cr_frame = self.frame[y_min:y_max, x_min:x_max]
            frame_hsv = cv.cvtColor(cr_frame, cv.COLOR_BGR2HSV)
        else:
            switch = False
        return switch, cr_frame, frame_hsv

    def __adjust_logo_color(self, required_field, decimals):
        """
        The method provides the adjustment of the logo's color relative to the insertion field

        :param required_field: field for the logo insertion
        :param decimals: the number of decimals to use when rounding the saturation parameter
        :return:
        """
        required_field = cv.cvtColor(required_field, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(required_field)
        mean_s = np.mean(s).astype(int)
        self.logo = cv.imread(self.logo)
        logo_hsv = cv.cvtColor(self.logo, cv.COLOR_BGR2HSV)
        logo_h, logo_s, logo_v = cv.split(logo_hsv)
        mean_logo_s = np.mean(logo_s).astype(int)
        s_coeff = round(mean_s / mean_logo_s, decimals)
        new_s_logo = (logo_s * s_coeff).astype('uint8')
        new_logo_hsv = cv.merge([logo_h, new_s_logo, logo_v])
        self.logo = cv.cvtColor(new_logo_hsv, cv.COLOR_HSV2BGR)

    def __detect_banner_color(self, frame_hsv, h_params, s_params, v_params):
        """
        The method provides the banners color detection and build the contour of the detected figure

        :param frame_hsv: transformed frame to the HSV mode
        :param h_params: hue parameters for detecting the required color
        :param s_params: saturation parameters for detecting required color
        :param v_params: value parameters for detecting required color
        :return:
        """
        low_color = np.array([h_params['low'], s_params['low'], v_params['low']])
        high_color = np.array([h_params['high'], s_params['high'], v_params['high']])
        color_mask = cv.inRange(frame_hsv, low_color, high_color)
        _, self.contours, __ = cv.findContours(color_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    def __find_diagonal_contour_coordinates(self, contour):
        """
        The method provides the detection of the diagonal contour coordinates for further computations

        :param contour: the contour of the required field for logo insertion
        :return: the list of diagonal contour coordinates
        """
        x_top_left = contour[:, 0, 0].min()
        x_bot_right = contour[:, 0, 0].max()
        y_top_left = contour[:, 0, 1].min()
        y_bot_right = contour[:, 0, 1].max()
        self.diagonal_coordinates_list = [x_top_left, x_bot_right, y_top_left, y_bot_right]
        return self.diagonal_coordinates_list

    def __find_contour_coordinates(self, cr_frame, deviation, area_threshold):
        """
        The method provides the detection of the contour corners coordinates

        :param cr_frame: field for the logo insertion
        :param deviation: the deviation parameter for tuning the corners coordinates
        :param area_threshold: the threshold for contour's area
        :return: banner mask, the left side of the contour for further computations, field for logo color adjustment
        """
        if len(self.contours) == 1:
            x_top_left, x_bot_right, y_top_left, y_bot_right = self.__find_diagonal_contour_coordinates(self.contours)
        else:
            coordinates_array = []
            for cnt in self.contours:
                area = cv.contourArea(cnt)
                if area > area_threshold:
                    x_top_left, x_bot_right, y_top_left, y_bot_right = self.__find_diagonal_contour_coordinates(cnt)
                    coordinates_array.append([x_top_left, x_bot_right, y_top_left, y_bot_right])
            x_top_left = min([coordinates_array[i][0] for i in range(len(coordinates_array))])
            x_bot_right = max([coordinates_array[i][1] for i in range(len(coordinates_array))])
            y_top_left = min([coordinates_array[i][2] for i in range(len(coordinates_array))])
            y_bot_right = max([coordinates_array[i][3] for i in range(len(coordinates_array))])

        self.diagonal_coordinates_list = [x_top_left, x_bot_right, y_top_left, y_bot_right]
        l_int = x_top_left + (x_bot_right - x_top_left) * deviation
        r_int = x_bot_right - (x_bot_right - x_top_left) * deviation

        min_x_max_y = []
        max_x_min_y = []

        for cnt in self.contours:
            area = cv.contourArea(cnt)
            if area > area_threshold:
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
        color_adj_field = cr_frame[y_top_left:y_left_side, x_top_left:x_bot_right]
        return banner_mask_cr, y_left_side, color_adj_field

    def __adjust_referee_colors(self, hsv_referee, area_threshold, frame_hsv, banner_mask_cr, coef,
                                hsv_body, hsv_flag):
        """
        The method provides referee colors adjustment for flowing around the detected banner

        :param hsv_referee: h, s, v parameters for referee object
        :param area_threshold: the threshold for the contours area
        :param frame_hsv: transformed frame to the HSV mode
        :param banner_mask_cr: banner mask
        :param coef: coefficients for referee object tuning
        :param hsv_body: h, s, v parameters for referee's body
        :param hsv_flag: h, s, v parameters for flag
        :return:
        """
        low_ref = np.array([hsv_referee['low_h'], 0, hsv_referee['low_v']])
        high_ref = np.array([hsv_referee['high_h'], 255, hsv_referee['high_v']])
        referee_mask = cv.inRange(frame_hsv, low_ref, high_ref)

        _, contours, _ = cv.findContours(referee_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > area_threshold[0]:
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
                low_bd = np.array([hsv_body['h'][0], hsv_body['s'][0], hsv_body['v'][0]])
                high_bd = np.array([hsv_body['h'][1], hsv_body['s'][1], hsv_body['v'][1]])
                body_mask = cv.inRange(ref_cr_hsv, low_bd, high_bd)

                _, contours, _ = cv.findContours(body_mask, cv.RETR_TREE,
                                                 cv.CHAIN_APPROX_SIMPLE)
                # drawing contour
                for cnt2 in contours:
                    area = cv.contourArea(cnt2)
                    if area > area_threshold[1]:
                        cv.drawContours(banner_mask_cr_ref, [cnt2], 0, (0, 255, 0), -1)

        # flag color
        low_flg = np.array([hsv_flag['h'][0], hsv_flag['s'][0], hsv_flag['v'][0]])
        high_flg = np.array([hsv_flag['h'][1], hsv_flag['s'][1], hsv_flag['v'][1]])
        flag_mask = cv.inRange(frame_hsv, low_flg, high_flg)

        _, contours, _ = cv.findContours(flag_mask, cv.RETR_TREE,
                                         cv.CHAIN_APPROX_SIMPLE)
        # drawing contour
        for cnt3 in contours:
            area = cv.contourArea(cnt3)
            if area > 1:
                cv.drawContours(banner_mask_cr, [cnt3], 0, (0, 255, 0), 1)

    def __resize_banner(self, y_left_side, resize_coef, w_threshold):
        """
        The method provides banner resizing

        :param y_left_side: the left side of the detected banner
        :param resize_coef: coefficient for banner width tuning
        :param w_threshold: width threshold
        :return: resized banner
        """
        top_left, bot_left, bot_right, top_right = self.coordinates_list
        x_top_left, x_bot_right, y_top_left, y_bot_right = self.diagonal_coordinates_list
        w = x_bot_right - x_top_left
        h = y_bot_right - y_top_left
        banner_height = y_left_side - y_top_left
        banner_width = w
        pred_width = int(banner_height * resize_coef)  # predicted width using proportions

        if w < w_threshold * pred_width:
            banner_width = pred_width
        xres = cv.resize(self.logo, (banner_width, h))  # resized banner

        rtx = [x_bot_right, y_bot_right - h]  # top right corner coordinates before transformation
        lbx = [x_bot_right - w, y_bot_right]  # left bottom corner coordinates before transformation

        pts1 = np.float32([[top_left, rtx, bot_right, lbx]]).reshape(-1, 1, 2)
        pts2 = np.float32([[top_left, top_right, bot_right, bot_left]]).reshape(-1, 1, 2)
        mtrx = cv.getPerspectiveTransform(pts1, pts2)
        resized_banner = cv.warpPerspective(xres, mtrx, (w, h), borderMode=1)
        return resized_banner

    def build_model(self, filename):
        """
        The method provides the ability to set the required parameters for model building

        :param filename: the file that contains required parameters for model tuning
        :return:
        """
        with open(filename, 'r') as stream:
            self.template_p = yaml.safe_load(stream)
        stream.close()

    def detect_banner(self):
        """
        The method provides the detection of the required field and prepares it for replacement

        :return: cropped field, resized banner, copy of cropped field, switch
        """
        switch, cr_frame, frame_hsv = self.__detect_contour(self.template_p['matcher'],
                                                            self.template_p['min_match_count'],
                                                            self.template_p['dst_threshold'],
                                                            self.template_p['nfeatures'],
                                                            self.template_p['neighbours'],
                                                            self.template_p['rc_threshold'])

        if switch:
            self.__detect_banner_color(frame_hsv, self.template_p['h_params'], self.template_p['s_params'],
                                       self.template_p['v_params'])

            banner_mask_cr, y_left_side, color_adjustment_field = self.__find_contour_coordinates(cr_frame,
                                                                  self.template_p['deviation'],
                                                                  self.template_p['cnt_area_threshold'])

            self.__adjust_logo_color(color_adjustment_field, self.template_p['decimals'])

            self.__adjust_referee_colors(self.template_p['hsv_referee'], self.template_p['area_threshold'], frame_hsv,
                                         banner_mask_cr, self.template_p['coef'], self.template_p['hsv_body'],
                                         self.template_p['hsv_flag'])

            resized_banner = self.__resize_banner(y_left_side, self.template_p['resize_coef'],
                                                  self.template_p['w_threshold'])

            return cr_frame, resized_banner, banner_mask_cr, switch
        else:
            return 0, 0, 0, switch

    def insert_logo(self, cr_frame, resized_banner, banner_mask_cr, switch):
        """
        The method provides the insertion of the required logo into the prepared field

        :param cr_frame: cropped field of frame
        :param resized_banner: resized banner
        :param banner_mask_cr: copy of cropped frame
        :param switch: indicates whether the required field was found or not
        :return:
        """
        if switch:
            for i in range(self.coordinates_list[0][1], self.coordinates_list[2][1] + 1):
                for j in range(self.coordinates_list[0][0], self.coordinates_list[2][0] + 1):
                    if list(banner_mask_cr[i, j]) == [0, 0, 255]:
                        cr_frame[i, j] = resized_banner[i - self.coordinates_list[0][1] - 1,
                                                        j - self.coordinates_list[0][0] - 1]
                    else:
                        continue
        else:
            pass
        cv.imshow('Replaced', self.frame)


if __name__ == '__main__':
    # set_visa_parameters()

    opencv_insertion = OpencvLogoInsertion('SET TEMPLATE', 'SET FRAME', 'SET LOGO')
    opencv_insertion.build_model('SET PARAMETERS')
    cropped_frame, resized_banner_, banner_mask_cr_, switch_ = opencv_insertion.detect_banner()
    opencv_insertion.insert_logo(cropped_frame, resized_banner_, banner_mask_cr_, switch_)

    # Press 'Esc' to close the resulting frame
    key = cv.waitKey(0)
    if key == 27:
        cv.destroyAllWindows()
