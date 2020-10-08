import json
import os

import cv2 as cv
import numpy as np


def load_image(path):
    return cv.imread(path)


def write_image(image, path):
    cv.imwrite(path, image)


def create_image(shape, color):
    image = np.ones(shape) * color
    return image.astype(np.uint8)


def binarize_image(image, blur_size, blur_sigma, otsu_threshold):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_blur = cv.GaussianBlur(image_gray, blur_size, blur_sigma)

    _, image_binary = cv.threshold(image_blur, otsu_threshold, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return image_binary


def canny(binary_image, threshold, alpha):
    return cv.Canny(binary_image, threshold, threshold * alpha)


def find_contours(binary_image):
    """
    returns a tuple of (contours, hierarchy)
    :param binary_image:
    :return: tuple => (contours, hierarchy)
    """
    return cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


def draw_contours(canvas, contours, hierarchy, color=(0, 0, 255), line_thickness=2):
    for i in range(len(contours)):
        cv.drawContours(canvas, contours, i, color, line_thickness, cv.LINE_AA, hierarchy, 0)


def draw_boxes(canvas, boxes, color=(0, 255, 0), line_thickness=2):
    for box in boxes:
        p1 = box[0]
        p2 = box[1]
        cv.rectangle(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, line_thickness)
        # it seems strange? yeah, its a bug on openCV side. the framework reports: "TypeError: an integer is required (got type tuple)"
        # if you write "cv.rectangle(canvas, box[0], box[1], color, line_thickness)"
        # despite the fact the provided argument must be a tuple of integers according to the docs.
        # this is the github issue in which i found a workaround for this problem : https://github.com/opencv/opencv/issues/14866


def draw_circles(canvas, circles, color=(255, 0, 0), line_thickness=1):
    for circle in circles:
        x, y, r = circle
        cv.circle(canvas, (x, y), r, color, line_thickness, cv.LINE_AA)


def get_circles_and_boxes(contours):
    """
    circles =>[ (x1, y1, r1), (x2, y2, r3), ... ] === (center_x, center_y, radius)
    boxes => [ ((x1, y1), (x'1, y'1), (w1, h1)) , ((x2, y2), (x'2, y'2), (w2, h2)), ... ] === ((x1, y1), (x2, y2), (height, width))
    :param contours:
    :return: circles and boxes
    """
    circles = []
    boxes = []
    for c in contours:
        _poly = cv.approxPolyDP(c, 3, True)
        _rect = cv.boundingRect(_poly)
        _center, _radius = cv.minEnclosingCircle(_poly)

        x1 = int(_rect[0])
        y1 = int(_rect[1])
        x2 = int(_rect[2])
        y2 = int(_rect[3])

        pt1 = (x1, y1)
        pt2 = (x1 + x2, y1 + y2)
        wh = (x2, y2)  # width and height

        boxes.append((pt1, pt2, wh))
        circle = (int(_center[0]), int(_center[1]), int(_radius))
        circles.append(circle)

    return np.array(circles), np.array(boxes)


def load_numpy_file(path):
    return np.load(path)


def write_numpy_array(array, path):
    np.save(path, array)


def load_json_file(path):
    with open(path, 'r') as file:
        json_data = json.load(file)
    return json_data


def file_exist(path):
    return os.path.isfile(path)


def dir_exist(path):
    return os.path.isdir(path)


def mkdir(path):
    os.makedirs(path)


def pad_image(image, new_size, pad_type="color", color=(0, 0, 0)):
    """
    adds padding and returns a new image with the given size
    :param image: original image
    :param new_size: a tuple comprise output width and height (h, w)
    :param color: color of the padded area
    :param pad_type: either "color" or "replicate"
    :return: padded image
    """
    img_h, img_w = image.shape[:2]

    h, w = new_size[0:2]

    delta_h = h - img_h
    delta_w = w - img_w

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    _border_type = cv.BORDER_CONSTANT if pad_type == "color" else cv.BORDER_REPLICATE

    return cv.copyMakeBorder(image, top, bottom, left, right, _border_type, value=color)


def resize_image(image, width=None, height=None):
    dim = None
    inter = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
        inter = cv.INTER_CUBIC if height > h else cv.INTER_AREA
    else:
        r = width / float(w)
        dim = (width, int(h * r))
        inter = cv.INTER_CUBIC if width > w else cv.INTER_AREA

    resized = cv.resize(image, dim, interpolation=inter)

    return resized


def resize_and_pad(image, new_size, pad_type="color", pad_color=(0, 0, 0)):
    (img_h, img_w) = image.shape[0:2]
    (new_h, new_w) = new_size[0:2]

    if img_w > img_h:
        output = resize_image(image, width=new_w)
    else:
        output = resize_image(image, height=new_h)

    output = pad_image(output, new_size, pad_type, pad_color)
    return output


def sample_image(image, crops, margin=0):
    images = []
    for crop in crops:
        ((x1, y1), (x2, y2)) = crop

        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = max(0, x2 + margin)
        y2 = max(0, y2 + margin)

        _img_crop = image[y1: y2, x1:x2]
        images.append(_img_crop)
    return images


def write_samples(samples, path):
    if not dir_exist(path):
        mkdir(path)
    for i in range(len(samples)):
        write_image(samples[i], F"{path}/{i}.jpg")
