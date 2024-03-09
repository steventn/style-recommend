import cv2
import numpy as np
import random
from webcolors import name_to_rgb, rgb_to_name


def generate_complementary_color(base_color):
    # Calculate the complementary color
    complementary_color = cv2.subtract(np.array([255, 255, 255], dtype=np.uint8), base_color)

    # Clip the values to ensure they are in the valid range (0-255)
    complementary_color = np.clip(complementary_color, 0, 255).astype(np.uint8)

    return complementary_color


def generate_analogous_colors(base_color, num_colors):
    # Calculate analogous colors by varying hue
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

def print_color_info(color_type, color_number, color_values):
    print(f"{color_type} Color {color_number}:")
    print(f"BGR {tuple(color_values)}")
    print(f"Name: {rgb_to_name(tuple(color_values))}")
    print()

def main():
    # Get the color name from the user
    color_name = input("Enter a color name: ")

    # Get the RGB value for the given color name
    base_color_rgb = name_to_rgb(color_name)
    base_color_bgr = np.array(base_color_rgb[::-1])  # Convert RGB to BGR

    # Display the information for the provided color
    print_color_info("Base", 1, base_color_bgr)

    # Generate and display complementary color
    complementary_color = generate_complementary_color(base_color_bgr)
    print_color_info("Complementary", 2, complementary_color)

    # Generate and display analogous colors
    num_analogous_colors = int(input("Enter the number of analogous colors to show: "))
    analogous_colors = generate_analogous_colors(base_color_bgr, num_analogous_colors)
    for i, color in enumerate(analogous_colors):
        print_color_info("Analogous", i + 3, color)

if __name__ == "__main__":
    main()
