import numpy as np
import cv2 as cv

four_cc = cv.VideoWriter_fourcc(*'MJPG')
out = cv.VideoWriter('/Users/oleksandr/Folder/WinStars/shape detect.avi', four_cc, 30, (1280, 720), True)

for i in range(0, 603):
    img = cv.imread('/Users/oleksandr/Folder/WinStars/old_frames/frame{}.png'.format(i))
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(gray_img, (5, 5), 0)
    # _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # thresh = cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, 2)
    # thresh = cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    # _, contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    corners = cv.goodFeaturesToTrack(thresh, 200, 0.01, 5)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv.circle(img, (x, y), 3, (0, 0, 255), -1)

    '''for cnt in contours:
        if cv.contourArea(cnt) > 200:
            cv.drawContours(img, [cnt], 0, (0, 255, 0), 2)'''

    cv.imshow('Contours', img)
    out.write(img)
    key = cv.waitKey(1)
    if key == 27:
        out.release()
        break
