import numpy as np
from PIL import Image
from numpy import asarray
import os
import time
import calendar


def combine_two_rows(top, bottom):
    answer = []
    row_width = len(top)

    for i in range(0, row_width, 2):
        top_left = top[i]
        top_right = top[i + 1]
        bottom_left = bottom[i]
        bottom_right = bottom[i + 1]

        curr_max = max(top_left, top_right, bottom_left, bottom_right)
        answer.append(curr_max)
    return answer


def reduce_image(input_image):
    reduced_image = []

    for i in range(0, len(input_image), 2):
        curr_row = input_image[i]
        next_row = input_image[i + 1]
        new_row = combine_two_rows(curr_row, next_row)
        reduced_image.append(new_row)

    return reduced_image


def get_gray(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])


# gmt_pre stores current gmtime - PRE
gmt_pre = time.gmtime()
print("gmt_pre:-", gmt_pre)

# gts stores timestamp
gts_pre = calendar.timegm(gmt_pre)
print("timestamp PRE:-", gts_pre)

asl_classes_path = r'<PATH_TO_DATASET>\asl-alphabet\asl_alphabet_train'
labels = os.listdir(asl_classes_path)
gray_path = r'<PATH_TO_OUTPUT>\output'

for curr_class in labels:

    # ORIGINAL IMAGE LOCATION
    curr_path = os.path.join(asl_classes_path, curr_class)
    curr_images = os.listdir(curr_path)

    # GRAYSCALE IMAGE LOCATION
    curr_gray_path = os.path.join(gray_path, curr_class)
    print('starting iteration for ', curr_path, 'curr_gray_path: ', curr_gray_path)
    os.mkdir(curr_gray_path)

    for curr_image in curr_images:
        curr_image_path = curr_path + '\\' + curr_image
        PIL_Image = Image.open(curr_image_path)
        img_array = asarray(PIL_Image)
        gray_image_array = get_gray(img_array)
        reduced_gray_image_array = reduce_image(gray_image_array)
        double_reduced_gray_image_array = reduce_image(reduced_gray_image_array)
        double_reduced_gray_np_array = asarray(double_reduced_gray_image_array)
        double_reduced_gray_image = Image.fromarray(double_reduced_gray_np_array)
        converted_gray_image = double_reduced_gray_image.convert("L")
        image_path = curr_gray_path + '\\' + curr_image
        converted_gray_image.save(image_path)


# gmt_post stores current gmtime - POST
gmt_post = time.gmtime()
print("gmt_post:-", gmt_post)

# gts stores timestamp
gts_post = calendar.timegm(gmt_post)
print("timestamp POST:-", gts_post)