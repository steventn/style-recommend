import cv2
import numpy as np
import random
import matplotlib.colors as colors
import webcolors


def generate_complementary_color(base_color):
    # Calculate the complementary color
    complementary_color = np.array([255, 255, 255]) - base_color
    return tuple(complementary_color)


def generate_monochromatic_colors(base_color, num_colors):
    hsv_base = cv2.cvtColor(np.uint8([[base_color]]), cv2.COLOR_BGR2HSV)[0][0]
    monochromatic_colors = []

    for _ in range(num_colors):
        # Generate random brightness values
        random_value = random.randint(0, 255)

        # Create a monochromatic color in HSV space
        monochromatic_hsv_color = np.array([hsv_base[0], hsv_base[1], random_value])

        # Convert back to BGR color space
        monochromatic_bgr_color = cv2.cvtColor(np.uint8([[monochromatic_hsv_color]]), cv2.COLOR_HSV2BGR)[0][0]

        monochromatic_colors.append(tuple(int(c) for c in monochromatic_bgr_color))

    return monochromatic_colors


def generate_analogous_colors(base_color, num_colors):
    hsv_base = cv2.cvtColor(np.uint8([[base_color]]), cv2.COLOR_BGR2HSV)[0][0]
    analogous_colors = []

    for _ in range(num_colors):
        # Generate random hue values within a range
        random_hue = random.randint(hsv_base[0] - 30, hsv_base[0] + 30) % 180

        # Create an analogous color in HSV space
        analogous_hsv_color = np.array([random_hue, hsv_base[1], hsv_base[2]])

        # Convert back to BGR color space
        analogous_bgr_color = cv2.cvtColor(np.uint8([[analogous_hsv_color]]), cv2.COLOR_HSV2BGR)[0][0]

        analogous_colors.append(tuple(int(c) for c in analogous_bgr_color))

    return analogous_colors


def color_name_to_bgr(color_name):
    try:
        rgb = colors.to_rgb(color_name)
        bgr = tuple(int(c * 255) for c in reversed(rgb))
        return bgr
    except ValueError:
        return None


def get_color_recommendations(color_name):
    bgr_color = color_name_to_bgr(color_name)
    base_color = np.array(bgr_color, dtype=np.uint8)
    color_dict = {}

    # Generate complementary color
    complementary_color = generate_complementary_color(base_color)
    color_dict["complementary_color"] = tuple(map(int, complementary_color))

    # Generate monochromatic colors
    num_monochromatic_colors = 4
    monochromatic_colors = generate_monochromatic_colors(base_color, num_monochromatic_colors)
    color_dict["monochromatic_colors"] = [tuple(map(int, color)) for color in monochromatic_colors]

    # Generate analogous colors
    num_analogous_colors = 4
    analogous_colors = generate_analogous_colors(base_color, num_analogous_colors)
    color_dict["analogous_colors"] = [tuple(map(int, color)) for color in analogous_colors]

    return color_dict


if __name__ == "__main__":
    get_color_recommendations("Brown")
