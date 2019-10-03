import numpy as np
import cv2
import sklearn
from joblib import load
import tensorflow as tf
import imutils
from math import floor, ceil
from scipy.spatial import distance as dist

#
# def process1(frame, model_svc, model_dnn, detector, threshold, threshold_value):
#     # cv2.imshow("frame", frame)
#
#
#     # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#     if frame is False:
#         return False
#
#     showed_img = frame.copy()
#     try:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     except:
#         return False
#     blur = cv2.GaussianBlur(gray, (5, 5), 1)
#
#     canny = imutils.auto_canny(blur)
#
#     dilated = cv2.dilate(canny.copy(), None, iterations=1).astype('int32')
#     dilated = np.uint8(dilated)
#
#
#     cnts = cv2.findContours(dilated.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0]
#
#     rois = np.ndarray((0, 28, 28, 1), dtype="uint8")
#     rects = []
#
#     for i, c in enumerate(cnts):
#         area = cv2.contourArea(c)
#         if area < 150:
#             continue
#
#         (x, y, w, h) = cv2.boundingRect(c)
#
#         if threshold:
#             if x + (w/2) < (frame.shape[1]*threshold_value/100):
#                 continue
#
#         roi = frame[y:y + h, x:x + w].copy()
#         # cv2.imshow('{}'.format(i), roi)
#
#         if not qualify(roi, model_svc):
#             continue
#         # to_save.append(roi)
#         cv2.drawContours(showed_img, [c], 0, (0, 255, 0), 1)
#         quant = subtract_background(roi)
#         # cv2.putText(showed_img, str(total), (int(x), int(y)),
#         #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#
#         rois = np.append(rois, [quant], axis=0)
#         rects.append([x, y, x + w, y + h])
#
#     predictions = model_dnn.predict([rois])
#
#     detector.update(rects, predictions)
#     results = detector.get_stats_result()
#     if threshold:
#         cv2.line(showed_img, (int(showed_img.shape[1]*threshold_value/100), 0),
#                  (int(showed_img.shape[1]*threshold_value/100), showed_img.shape[0]), (255, 255, 0), 1)
#
#     total = 0
#     for objectID, centroid in detector.objects.items():
#         cv2.putText(showed_img, str(np.argmax(results[objectID])),
#                     (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
#                     cv2.LINE_AA)
#         total += int(np.argmax(results[objectID]))
#
#     return (total, showed_img)

def process(frame, model_svc, model_dnn, detector, threshold, threshold_value):
    # cv2.imshow("frame", frame)

    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if frame is False:
        return False

    showed_img = frame.copy()
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except:
        return False
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    diff = cv2.subtract(gray, blur)
    # cv2.imshow('gray', diff)
    blur2 = cv2.GaussianBlur(diff * 50, (3, 3), 1)
    thresh = cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 5)

    dilated = cv2.dilate(thresh.copy(), None, iterations=1).astype('int32')
    dilated = np.uint8(dilated)


    cnts = cv2.findContours(dilated.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]

    rois = np.ndarray((0, 28, 28, 1), dtype="uint8")
    rects = []

    for i, c in enumerate(cnts):
        area = cv2.contourArea(c)
        if area < 300:
            continue

        (x, y, w, h) = cv2.boundingRect(c)

        if threshold:
            if x + (w/2) < (frame.shape[1]*threshold_value/100):
                continue

        roi = frame[y:y + h, x:x + w].copy()
        # cv2.imshow('{}'.format(i), roi)

        if not qualify(roi, model_svc):
            continue
        # to_save.append(roi)
        cv2.drawContours(showed_img, [c], 0, (0, 255, 0), 1)
        quant = subtract_background(roi)
        # cv2.putText(showed_img, str(total), (int(x), int(y)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        rois = np.append(rois, [quant], axis=0)
        rects.append([x, y, x + w, y + h])

    predictions = model_dnn.predict([rois])

    detector.update(rects, predictions)
    results = detector.get_stats_result()
    if threshold:
        cv2.line(showed_img, (int(showed_img.shape[1]*threshold_value/100), 0),
                 (int(showed_img.shape[1]*threshold_value/100), showed_img.shape[0]), (255, 255, 0), 1)

    total = 0
    for objectID, centroid in detector.objects.items():
        cv2.putText(showed_img, str(np.argmax(results[objectID])),
                    (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)
        total += int(np.argmax(results[objectID]))

    return (total, showed_img)

def detect_corners(img):
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]

    canny = cv2.Canny(v, 75, 200)
    cnts = cv2.findContours(canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            # show the contour (outline)
            # of the piece of paper if debug is True

            # apply the four point transform to obtain a top-down
            # view of the original image
            approx = approx.reshape(4, 2)

            return approx


    return False

def transform_sheet(img, approx):
    warped = four_point_transform(img.copy(), approx)

    # Determine the ratio and detect if it is approximately the same one as a sheet.
    # test for both portrait and landscape scanning
    ratio = warped.shape[0] / warped.shape[1]
    if (((ratio > 0.5) and (ratio < 1)) or (
            (1 / ratio > 0.5) and (1 / ratio < 1))):
        try:
            warped = cv2.resize(warped, (int(warped.shape[0] * 0.77), int(warped.shape[0])))
        except:
            return False

        return warped.copy()[5:warped.shape[0] - 5, 5:warped.shape[1] - 5]
    return False


def subtract_background(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    (T, thresh) = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    quant = imutils.resize(thresh, width=None if np.argmax(thresh.shape) == 0 else 28,
                           height=None if np.argmax(thresh.shape) == 1 else 28)
    quant = cv2.copyMakeBorder(quant,
                               0 if np.argmax(quant.shape) == 0 else floor((28 - quant.shape[0]) / 2),
                               0 if np.argmax(quant.shape) == 0 else ceil((28 - quant.shape[0]) / 2),
                               0 if np.argmax(quant.shape) == 1 else floor((28 - quant.shape[1]) / 2),
                               0 if np.argmax(quant.shape) == 1 else ceil((28 - quant.shape[1]) / 2),
                               cv2.BORDER_CONSTANT, value=0)
    quant = quant.reshape(28, 28, 1)
    # normalized_rois = tf.keras.utils.normalize([quant], axis=1)
    # print(normalized_rois)
    quant = quant.astype('float32')
    quant /= 255
    return quant
    # predictions = model.predict([[quant]])
    # return np.argmax(predictions[0])


def qualify(img, model):
    lower_red = np.array([40, 55, 115])
    upper_red = np.array([95, 101, 150])


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 5)

    background_removed = cv2.bitwise_and(img, img, mask=thresh)


    average = cv2.mean(img, thresh)
    avg_patch = np.ones(shape=img.shape, dtype=np.uint8) * np.uint8(average[:3])

    result = model.predict([average[:3]])
    if result == [1]:
        return True
    else:
        return False


def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(image, rect):
    # obtain a consistent order of the points and unpack them
    # individually
    # rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (maxWidth, maxHeight))

    # return the warped image
    return warped
