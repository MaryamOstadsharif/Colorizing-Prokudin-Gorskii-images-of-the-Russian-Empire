import numpy as np
import cv2
import os


# Function to set paths based on the resolution level
def set_path(level):
    if level == 'Low_resolution':
        # Path for low-resolution images
        data_path = r'E:\Computer vision\ex1\hw1\data_1/'.replace('\\', '/')
    else:
        # Path for high-resolution images
        data_path = r'E:\Computer vision\ex1\hw1\data_2/'.replace('\\', '/')

    # Common result path
    result_path = r'E:\Computer vision\HW1\result'.replace('\\', '/')

    return data_path, result_path


# Function to read image filenames from a directory
def read_img_filename(path):
    img_file_names = os.listdir(path)  # List all files in the directory
    return img_file_names


# Function to convert 16-bit image to 8-bit by scaling pixel values
def convert_uint16_to_uint8(img_uint16):
    img_uint8 = (img_uint16 / 65535.0 * 255).astype(np.uint8)
    return img_uint8


# Function to split an image into its blue, green, and red channels
def split_to_rgb(img):
    h = int(img.shape[0] / 3)
    # Split the image into blue, green, and red channels
    blue, green, red = [img[h * i:h * (i + 1)] for i in range(3)]
    return blue, green, red


# Function to crop the image by removing 10% from each side
def crop(img):
    h, w = img.shape
    new_height = [int(h * 0.1), img.shape[0] - int(h * 0.1)]  # Define new height bounds after cropping
    new_weight = [int(w * 0.1), img.shape[1] - int(w * 0.1)]  # Define new width bounds after cropping
    return img[new_weight[0]:new_height[1], new_weight[0]:new_weight[1]]


# Function to calculate the cross-correlation between two images
def cross_correlation(m, n):
    m_norm = m / np.linalg.norm(m)
    n_norm = n / np.linalg.norm(n)
    # Calculate the cross-correlation
    cc = np.sum(m_norm * n_norm)
    return cc


# Function to find the best shift between two images using cross-correlation
def find_shift(img1, img2, bound, radius):
    best_score = -1
    # Iterate over possible shifts within the radius around the bound
    for column in range(-radius + bound[0], radius + bound[0]):
        for row in range(-radius + bound[1], radius + bound[1]):
            # Calculate the cross-correlation for the shifted image
            img_shift_score = cross_correlation(img1, np.roll(np.roll(img2, column, axis=0), row, axis=1))
            # Update the best score and shift if a better match is found
            if img_shift_score > best_score:
                best_score = img_shift_score
                img_shift = [column, row]
    return img_shift


# Function to apply a shift to an image and align it
def align(img, shift):
    aligned_img = np.roll(np.roll(img, shift[0], axis=0), shift[1], axis=1)
    return aligned_img


# Function to perform simple alignment using a fixed radius
def simple_align(img1, img2, radius):
    bound = [radius, radius]  # Define the bounds for shifting
    # Find the best shift using cross-correlation
    shift = find_shift(img1, img2, bound, radius)
    return shift


# Function to perform multi-scale alignment for large images
def multi_scale_align(img1, img2, radius):
    rows, columns = img1.shape
    if max(rows, columns) > 200:
        # Resize both images to half of their original size
        img1_resized = cv2.resize(img1, (int(columns * 0.5), int(rows * 0.5)))
        img2_resized = cv2.resize(img2, (int(columns * 0.5), int(rows * 0.5)))

        # Perform multi-scale alignment on the resized images (coarser scale)
        courser_image_offset = multi_scale_align(img1_resized, img2_resized, radius)

        # Scale the coarse offset by 2 to adjust for the image resizing
        scaled_courser_offset = [x * 2 for x in courser_image_offset]

        # Find the finer shift using the full-sized images and the scaled offset
        shift = find_shift(img1, img2, scaled_courser_offset, radius)
    else:
        # For small images, perform simple alignment with a fixed radius of 15
        shift = simple_align(img1, img2, 15)

    return shift
