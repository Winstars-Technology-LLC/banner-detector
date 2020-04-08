import cv2 as cv
import numpy as np
import math
import pandas as pd


def detect_lines(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # blur_gray = cv.GaussianBlur(gray, (3, 3), 0)

    # _, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    # th = cv.adaptiveThreshold(blur_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    # laplacian = cv.Laplacian(gray, cv.CV_64F, ksize=3)
    # laplacian = np.uint8(np.absolute(laplacian))

    # sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    # sobely = np.uint8(np.absolute(sobely))

    low_threshold = 30
    high_threshold = 60

    low_color = np.array([35, 85, 60])  # 35 25 25
    high_color = np.array([50, 255, 255])  # 70 255 255
    color_mask = cv.inRange(hsv, low_color, high_color)
    _, contours, __ = cv.findContours(color_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    drop_list = [i for i in range(len(contours)) if cv.contourArea(contours[i]) < 2500]
    contours = [i for j, i in enumerate(contours) if j not in drop_list]

    min_y = np.min([[np.min([point[0][1] for point in contours[i]])] for i in range(len(contours))])

    endpoints = []
    for cnt in contours:
        for point in cnt:
            if point[0][1] == min_y:
                endpoints.append(point[0])

    min_x = np.min([point[0] for point in endpoints])
    max_x = image.shape[1] - 1

    max_y = []
    for i in range(len(contours)):
        for point in contours[i]:
            if point[0][0] == max_x:
                max_y.append(point[0][1])

    max_y = np.min(max_y)

    base_line_tangent = (max_y - min_y)/(max_x - min_x)
    base_line_angle = math.degrees(math.atan(base_line_tangent))

    # for cnt in contours:
    #     cv.drawContours(image, [cnt], -1, (0, 255, 0), 2)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180
    threshold = 200  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 200
    max_line_gap = 60  # maximum gap in pixels between connectible line segments
    line_image = np.copy(image) * 0

    cv.circle(line_image, (min_x, min_y), 5, (0, 0, 255), 2)
    cv.circle(line_image, (max_x, max_y), 5, (0, 0, 255), 2)

    cv.line(line_image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

    edges = cv.Canny(gray, low_threshold, high_threshold)

    lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                           min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:

            line_tangent = (y2 - y1) / (x2 - x1)
            line_angle = math.degrees(math.atan(line_tangent))

            # dst = abs((y2 - y1) * base_line_center[0] - (x2 - x1) * base_line_center[1] + x2 * y1 - y2 * x1) /\
                      # np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

            # if abs(line_angle - base_line_angle) <= 0.15:
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            cv.line(line_image, (x1, y1), (x2, y2), color, 2)

    lines_edges = cv.addWeighted(image, 0.8, line_image, 1, 0)
    return lines_edges


def build_model(video, img_name='Set name', files_folder_path='Set path'):
    if not video:
        img = cv.imread(files_folder_path + img_name)
        lines = detect_lines(img)
        cv.imshow('result', lines)
        cv.imwrite(files_folder_path + 'single_line.png', lines)
        key = cv.waitKey(0)
        if key == 27:
            cv.destroyAllWindows()
    else:
        capture = cv.VideoCapture(files_folder_path + 'football.mp4')
        frame_width = int(capture.get(3))
        frame_height = int(capture.get(4))
        four_cc = cv.VideoWriter_fourcc(*'MJPG')
        # out = cv.VideoWriter(files_folder_path + 'lines no filter angle.avi',
                             # four_cc, 30, (frame_width, frame_height), True)
        while capture.isOpened():
            res, frame = capture.read()
            if res:
                lines = detect_lines(frame)
                cv.imshow('result', lines)
                # out.write(lines)
                key = cv.waitKey(1)
                if key == 27:
                    break
            else:
                break

        capture.release()
        # out.release()


folder = '/Users/oleksandr/Folder/WinStars/'
image_name = 'frame10.png'
build_model(True, image_name, folder)
