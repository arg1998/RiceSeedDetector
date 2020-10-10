import numpy as np
import cv2 as cv
import utils

STD_THRESHOLD_BLUE = 1.2 # STD coefficient for blue image
CROP_SIZE = 400 # image crop size
# respective configs for blue and black bg images 
CONFIGS = {
    "blue": {
        "otsu_threshold": 120,
        "canny_threshold": 120,
        "canny_alpha": 1.6,
        "blur_size": (25, 25),
        "blur_sigma": 25
    },
    "black": {
        "otsu_threshold": 125,
        "canny_threshold": 120,
        "canny_alpha": 1.6,
        "blur_size": (3, 3),
        "blur_sigma": 3
    }
}

# image file configs 
IMAGE_DATASET_INFO = [
    {
        "background_color": "blue",
        "name": "blue_bg",
        "extension": "jpg"
    },
    {
        "background_color": "black",
        "name": "black_bg",
        "extension": "png"
    }

]

for image_entry in IMAGE_DATASET_INFO:
    _img_extension = image_entry["extension"]
    _img_file_name = image_entry["name"]
    _img_background_color = image_entry["background_color"]
    _img_cfg = CONFIGS[_img_background_color]

    # load image 
    original_image = utils.load_image(F"input/{_img_file_name}.{_img_extension}")
    # producing a binary image (OTSU thresholding) after converting the original image into a grey-scale image
    binary_image = utils.binarize_image(original_image, _img_cfg["blur_size"], _img_cfg["blur_sigma"], _img_cfg["otsu_threshold"])
    # extracting edges from previously generated binary image, using canny edge detection algorithm 
    canny_image = utils.canny(binary_image, _img_cfg["canny_threshold"], _img_cfg["canny_alpha"])
    # calculating outlines of grains by finding their contours (extracting the most external contour in each hierarchy)
    _, contours, hierarchy = utils.find_contours(canny_image)
    # creating a white background image to use as a canvas 
    white_canvas = utils.create_image((*canny_image.shape, 3), 255)
    # drawing grains outlines over the canvas 
    utils.draw_contours(white_canvas, contours, hierarchy)
    # calculating circumferencing circles and boxes for each contour 
    circles, boxes = utils.get_circles_and_boxes(contours)

    contour_image = white_canvas.copy()
    # drawing enclosing circles and rectangles
    utils.draw_circles(white_canvas, circles)
    utils.draw_boxes(white_canvas, boxes)

    # ------------------------- filtering small and large objects ------------------------------------
    # filtering small and large objects using median and standard deviation over objects radii as a measure
    _radius = circles[:, 2]  # extract the radii from detected objects' circular boundaries 
    _std = np.std(_radius)
    _median = np.median(_radius)

    _refined_circles = None
    _ignored_circles = None
    _mask = None
    # filtering conditions 
    if _img_background_color == "black":
        _mask = (_median + 25 > _radius) & (_radius > _median - 25)
    else:
        _mask = _radius > _median - _std

    # applying mask indices over unfilterd objects
    _refined_circles = circles[_mask]
    _ignored_circles = circles[np.bitwise_not(_mask)]

    refined_image = contour_image.copy()
    utils.draw_circles(refined_image, _ignored_circles)
    utils.draw_circles(refined_image, _refined_circles, (0, 255, 0), 2)

    crop_image = contour_image.copy()
    _margin = np.int32(CROP_SIZE / 2)
    _circle_centers = _refined_circles[:, 0:2].astype(np.int32)
    # calculating crop region coordinations based on _margin
    p1_x = np.subtract(_circle_centers[:, 0], _margin, dtype=np.int32)
    p1_y = np.subtract(_circle_centers[:, 1], _margin, dtype=np.int32)

    p2_x = np.add(_circle_centers[:, 0], _margin, dtype=np.int32)
    p2_y = np.add(_circle_centers[:, 1], _margin, dtype=np.int32)

    pt1 = np.array(tuple(zip(p1_x, p1_y)), dtype=np.int32)
    pt2 = np.array(tuple(zip(p2_x, p2_y)), dtype=np.int32)

    _image_crop_area = list(zip(pt1, pt2))
    # drawing crop regions
    utils.draw_boxes(crop_image, _image_crop_area)

    utils.write_image(binary_image, F"output/{_img_background_color}/01 - binary.{_img_extension}")
    utils.write_image(canny_image, F"output/{_img_background_color}/02 - canny.{_img_extension}")
    utils.write_image(contour_image, F"output/{_img_background_color}/03 - contours.{_img_extension}")
    utils.write_image(white_canvas, F"output/{_img_background_color}/04 - boundaries.{_img_extension}")
    utils.write_image(refined_image, F"output/{_img_background_color}/05 - refined.{_img_extension}")
    utils.write_image(crop_image, F"output/{_img_background_color}/06 - crops.{_img_extension}")

    # saving boundaries in a numpy file for future calculations 
    utils.write_numpy_array(circles, F"output/{_img_background_color}/{_img_file_name}.circles.npy")
    utils.write_numpy_array(boxes, F"output/{_img_background_color}/{_img_file_name}.boxes.npy")
