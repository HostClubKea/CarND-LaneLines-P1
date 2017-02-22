#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import math


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    #return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #coefficients = [0.4, 0.4, 0.2]
    #m = np.array(coefficients).reshape((1,3))
    #return cv2.transform(img, m)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    # in the task we need to find 2 lanes, we know that one is on the right and one is on the left
    # we don't need to check all lines - horizontal and lines with some treshhold slope could be removed

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    return lines



def lane_lines(lines, width, height, threshold_slope = 0.5, upper_cutoff = 0.65):
    #Filter hough lines and constract 2 lane lines
    left_points = []
    right_points = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 != x2:
                slope = ((y2 - y1) / (x2 - x1))
                if abs(slope) > threshold_slope:
                    line_length = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                    if slope > 0 and x1 > width // 2:
                        right_points += [[x1, y1, int(line_length), x2, y2], [x2, y2, int(line_length), x1, y1]]
                    elif slope < 0 and x1 < width // 2:
                        left_points += [[x1, y1, int(line_length), x2, y2], [x2, y2, int(line_length), x1, y1]]


    #Tried to filter outliers from hough lanes by length, didn't helped
    right_points = sorted(right_points, key=lambda x: x[2])
    left_points = sorted(left_points, key=lambda x: x[2])

    right_points = np.array(right_points)#[:20])
    left_points = np.array(left_points)#[:20])

    filtered_lines = []
    for x1, y1, l, x2, y2 in left_points:
        filtered_lines += [[[x1, y1, x2, y2]]]

    for x1, y1, l, x2, y2 in right_points:
        filtered_lines += [[[x1, y1, x2, y2]]]

    #In chalange video sometimes there wasn't any lines
    if right_points.size == 0 or left_points.size == 0:
        return [], filtered_lines

    #Now we need to find 2 lane lines from right and left set of points
    r_slope, r_b = np.polyfit(right_points[:, 0], right_points[:, 1], 1)
    l_slope, l_b = np.polyfit(left_points[:, 0], left_points[:, 1], 1)

    cutoff = int(upper_cutoff * height)

    #Find right and left lane points from slope and b
    left_line = [int((cutoff - l_b) / l_slope), cutoff, int((height - l_b) / l_slope), height]
    right_line = [int((cutoff - r_b) / r_slope), cutoff, int((height - r_b) / r_slope), height]

    lanes = [[left_line], [right_line]]
    return lanes, filtered_lines

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def smooth_lane(prev_lines, lines, smooth_window):
    #Smooth lane lines based on previous lane lines
    if len(lines) > 0:
        prev_lines.insert(0,lines)#append(lines)

    prev_lines = np.squeeze(prev_lines)
    line_l = [0, 0, 0, 0]
    line_r = [0, 0, 0, 0]

    size = len(prev_lines[:smooth_window])
    for line in prev_lines[:smooth_window]:
        line_l += line[0]
        line_r += line[1]

    line_l = line_l / size
    line_r = line_r / size

    return np.array([[line_l, line_r]], dtype=np.int32)


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def detect_lines(image, prev_lines = [], smooth=False, smooth_window = 5, convert_grayscale=True, kernel_size=5, low_threshold=70, high_threshold=200, rho=2, theta=1, threshold=20, min_line_length=5, max_line_gap=5):
    #Function  represents base pipeline and returns lane lines and hough lines and filtered hough lines
    #This was made for easier testing
    theta = theta * np.pi / 180

    #Get image shape info
    imshape = image.shape
    width = imshape[1]
    height = imshape[0]

    # Get graysacale image from original
    if convert_grayscale:
        gray = grayscale(image)
    else:
        gray = image

    # Apply Gaussian smoothing to grayscale image
    blur_gray = gaussian_blur(gray, kernel_size)

    #fix. otherwise was exception
    blur_gray_copy = np.uint8(blur_gray)

    # Detext edges
    edges = canny(blur_gray_copy, low_threshold, high_threshold)


    vertices = np.array([[(50, height), (width / 2 - 20, height / 2 + 50),
                          (width / 2 + 20, height / 2 + 50), (width - 50, height)]], dtype=np.int32)

    #Apply region mask
    region = region_of_interest(edges, vertices)

    h_lines = hough_lines(region, rho, theta, threshold, min_line_length, max_line_gap)

    lanes, filtered_h_lines = lane_lines(h_lines, imshape[1], imshape[0])

    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    if smooth:
        s_lanes = smooth_lane(prev_lines, lanes, smooth_window)
        draw_lines(line_img, s_lanes, [255, 0, 0], 7)
    else:
        draw_lines(line_img, lanes, [255, 0, 0], 7)


    #draw_lines(line_img, h_lines, [0, 255, 0], 2)
    #draw_lines(line_img, filtered_h_lines, [0, 0, 255], 2)
    #draw_lines(line_img, lanes, [255, 0, 0], 7)

    lines_edges = weighted_img(initial_img=image, img=line_img)

    return lines_edges


prev_lines = []

def process_image2(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    return detect_lines(image)


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    return detect_lines(image, prev_lines=prev_lines, smooth=True,  convert_grayscale=False, kernel_size=7, low_threshold=50, high_threshold=150, rho=2, theta=1, threshold=50, min_line_length=20, max_line_gap=20)

challenge_output = 'extra_9.mp4'
clip2 = VideoFileClip('challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)

white_output = 'white_2.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image2) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

yellow_output = 'yellow_2.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image2)
yellow_clip.write_videofile(yellow_output, audio=False)