from utils import *
import time
from PIL import Image
import matplotlib.pyplot as plt

# Set the resolution level for images ('Low_resolution' or 'High_resolution')
img_level = 'High_resolution'

# Get the corresponding data and result paths based on the image resolution level
data_path, result_path = set_path(img_level)

# Read the image filenames from the data directory
img_names = read_img_filename(data_path)

# Loop through the images
for img_curr in img_names:
    print(f'\n ******************* Processing image: {img_curr} **********************')

    # Start tracking the processing time
    start_time = time.time()

    # Load the image and convert it to a NumPy array
    orig_img = np.array(Image.open(data_path + img_curr))

    # If the image is high resolution, convert it from 16-bit to 8-bit
    if img_level == 'High_resolution':
        orig_img = convert_uint16_to_uint8(orig_img)

    # Split the image into blue, green, and red channels
    blue_img, green_img, red_img = split_to_rgb(orig_img)

    # Crop the individual color channels
    blue_img = crop(blue_img)
    green_img = crop(green_img)
    red_img = crop(red_img)

    # Create a list of the RGB channels for visualization
    images_rgb = [blue_img, green_img, red_img]

    # Create a subplot to display each color channel side by side
    fig, axs = plt.subplots(1, len(images_rgb), figsize=(15, 5))

    # Display each color channel in the subplot
    for i, ax in enumerate(axs):
        ax.imshow(images_rgb[i])
        ax.axis('off')  # Hide the axes for a cleaner display

    # Save the subplot as a jpg file
    plt.savefig(f'{result_path}/{img_level}/{img_curr.split(".")[0]}_subplot.jpg', bbox_inches='tight')

    # Align the color channels based on the resolution level
    if img_level == 'Low_resolution':
        # Simple alignment for low-resolution images
        shift_green = simple_align(blue_img, green_img, radius=7)
        aligned_green = align(green_img, shift_green)
        shift_red = simple_align(blue_img, red_img, radius=7)
        aligned_red = align(red_img, shift_red)
    else:
        # Multi-scale alignment for high-resolution images
        shift_green = multi_scale_align(blue_img, green_img, radius=7)
        aligned_green = align(green_img, shift_green)
        shift_red = multi_scale_align(blue_img, red_img, radius=7)
        aligned_red = align(red_img, shift_red)

    # Print the computed shifts for green and red channels
    print(f"\n ####### Shift: \n Green channel= {shift_green} \n Red channel= {shift_red}")

    # Stack the aligned red, green, and blue channels to create a color image
    color_img = np.dstack([aligned_red, aligned_green, blue_img])

    # Save the final aligned color image
    plt.imsave(f'{result_path}/{img_level}/{img_curr.split(".")[0]}.jpg', color_img)

    # Stop the time and print the total processing time for the image
    end_time = time.time()
    print(f'\n Total time: {round((end_time - start_time), 4)} seconds')
