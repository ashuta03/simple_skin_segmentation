import cv2
import numpy as np
import os
import sys
import argparse


def hsv_filter(im):
    """
    using opencv inRange to filter out skin color from hsv color space of the video
    using consecutive morphology operations to reduce noisy segmentation - false positives and false negatives

    Args:
        im: np.array

    Returns:
        in_range: np.array
        masked_img: np.array

    """
    lower_limit = np.array([0, 35], np.uint8)
    upper_limit = np.array([30, 255], np.uint8)

    blurred = cv2.GaussianBlur(im, (17, 17), 3)

    hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    in_range = cv2.inRange(hsv_image[:, :, 0:2], lower_limit, upper_limit)
    in_range = cv2.GaussianBlur(in_range, (3, 3), 2)

    kernel_1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    opened_mask = cv2.morphologyEx(in_range, cv2.MORPH_OPEN, kernel_1)
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel_2)

    masked_img = cv2.bitwise_and(im, im, mask=closed_mask)

    return in_range, masked_img


def fix_brightness(im):
    """
    Adjusts brightness and contrast of the image
    Args:
        im: np.array

    Returns:
        im: np.array

    """
    brightness = 50
    contrast = 40

    im = np.int16(im)
    im = im * (contrast / 127 + 1) - contrast + brightness
    im = np.clip(im, 0, 255)
    im = np.uint8(im)
    return im


def video_display(video_file):
    """

    Args:
        video_file: str or int

    Returns:

    """
    camera = cv2.VideoCapture(video_file)
    while True:
        try:
            grabbed, frame = camera.read()
            frame = fix_brightness(frame)
            seg, skin = hsv_filter(frame)
            cv2.imshow('original_frame', frame)
            cv2.imshow('segmented_mask', seg)
            cv2.imshow('segmented_skin', skin)
            cv2.waitKey(3)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except IndexError as e:
            print(e)
            continue

        except BaseException as e:
            print(e)
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HSV skin segmentation')
    parser.add_argument('--video', '-t', default='live', type=str, help='json config file path')

    args = parser.parse_args()

    video = args.video

    if video == 'live':
        video = 0

    else:
        if not os.path.isfile(video) or os.path.splitext(video)[1] not in ['.avi', '.mp4']:
            print('Not a video file')
            sys.exit(0)

    video_display(video)