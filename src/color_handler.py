import cv2
import numpy as np
import random
import matplotlib.colors as colors
import webcolors


def generate_complementary_color(base_color):
    # Calculate the complementary color
    complementary_color = np.array([255, 255, 255], dtype=np.uint8) - base_color
    return complementary_color


def generate_monochromatic_colors(base_color, num_colors):
    hsv_base = cv2.cvtColor(np.uint8([[base_color]]), cv2.COLOR_BGR2HSV)[0][0]
    monochromatic_colors = []

    for _ in range(num_colors):
        # Generate random brightness values
        random_value = random.randint(0, 255)

        # Create a monochromatic color in HSV space
        monochromatic_hsv_color = np.array([hsv_base[0], hsv_base[1], random_value], dtype=np.uint8)

        # Convert back to BGR color space
        monochromatic_bgr_color = cv2.cvtColor(np.uint8([[monochromatic_hsv_color]]), cv2.COLOR_HSV2BGR)[0][0]

        monochromatic_colors.append(monochromatic_bgr_color)

    return monochromatic_colors


def generate_analogous_colors(base_color, num_colors):
    hsv_base = cv2.cvtColor(np.uint8([[base_color]]), cv2.COLOR_BGR2HSV)[0][0]
    analogous_colors = []

    for _ in range(num_colors):
        # Generate random hue values within a range
        random_hue = random.randint(hsv_base[0] - 30, hsv_base[0] + 30) % 180

        # Create an analogous color in HSV space
        analogous_hsv_color = np.array([random_hue, hsv_base[1], hsv_base[2]], dtype=np.uint8)

        # Convert back to BGR color space
        analogous_bgr_color = cv2.cvtColor(np.uint8([[analogous_hsv_color]]), cv2.COLOR_HSV2BGR)[0][0]

        analogous_colors.append(analogous_bgr_color)

    return analogous_colors


def color_name_to_bgr(color_name):
    try:
        rgb = colors.to_rgb(color_name)
        bgr = tuple(int(c * 255) for c in reversed(rgb))
        return bgr
    except ValueError:
        return None


def bgr_to_color_name(bgr):
    # Convert BGR to RGB
    rgb = bgr[::-1]
    # Convert RGB to hex
    hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb)
    # Find the closest color name
    try:
        color_name = webcolors.hex_to_name(hex_color)
        return color_name
    except ValueError:
        # If the color name is not recognized, return None
        return None


def get_color_recommendations(color_name):
    bgr_color = color_name_to_bgr(color_name)
    base_color = np.array(bgr_color, dtype=np.uint8)

    # Generate complementary color
    complementary_color = generate_complementary_color(base_color)
    print("Complementary color:", complementary_color)

    # Generate monochromatic colors
    num_monochromatic_colors = 4
    monochromatic_colors = generate_monochromatic_colors(base_color, num_monochromatic_colors)
    print("Monochromatic colors:")
    for i, color in enumerate(monochromatic_colors):
        print(f"Color {i + 1}: {color}")

    # Generate analogous colors
    num_analogous_colors = 4
    analogous_colors = generate_analogous_colors(base_color, num_analogous_colors)
    print("Analogous colors:")
    for i, color in enumerate(analogous_colors):
        print(f"Color {i + 1}: {color}")


def main():
    # Input a color in BGR format
    base_color = np.array(input("Enter the base color (BGR format, e.g., 255 0 0 for red): ").split(), dtype=np.uint8)

    # Generate complementary color
    complementary_color = generate_complementary_color(base_color)
    print("Complementary color:", complementary_color)

    # Generate monochromatic colors
    num_monochromatic_colors = int(input("Enter the number of monochromatic colors to generate: "))
    monochromatic_colors = generate_monochromatic_colors(base_color, num_monochromatic_colors)
    print("Monochromatic colors:")
    for i, color in enumerate(monochromatic_colors):
        print(f"Color {i + 1}: {color}")

    # Generate analogous colors
    num_analogous_colors = int(input("Enter the number of analogous colors to generate: "))
    analogous_colors = generate_analogous_colors(base_color, num_analogous_colors)
    print("Analogous colors:")
    for i, color in enumerate(analogous_colors):
        print(f"Color {i + 1}: {color}")


if __name__ == "__main__":
    # main()
    get_color_recommendations("Brown")
