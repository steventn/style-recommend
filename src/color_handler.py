import json

import cv2
import numpy as np
import random
import matplotlib.colors as colors
import requests


def get_color_scheme(rgb_code, mode):
    url = f"https://www.thecolorapi.com/scheme?rgb={rgb_code}&mode={mode}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: Unable to fetch color scheme. Status code: {response.status_code}")
        return None


def get_color_scheme_complementary(color_name):
    rgb = colors.to_rgb(color_name)
    color_scheme = get_color_scheme(rgb, "complement")
    rgb_colors = [color['rgb']['value'] for color in color_scheme['colors']]
    rgb_tuples = [tuple(map(int, color.strip('rgb()').split(','))) for color in rgb_colors]
    return rgb_tuples


def get_color_scheme_monochromatic(color_name):
    rgb = colors.to_rgb(color_name)
    color_scheme = get_color_scheme(rgb, "monochrome")
    rgb_colors = [color['rgb']['value'] for color in color_scheme['colors']]
    rgb_tuples = [tuple(map(int, color.strip('rgb()').split(','))) for color in rgb_colors]
    return rgb_tuples


def get_color_scheme_analogous(color_name):
    rgb = colors.to_rgb(color_name)
    color_scheme = get_color_scheme(rgb, "analogic")
    rgb_colors = [color['rgb']['value'] for color in color_scheme['colors']]
    rgb_tuples = [tuple(map(int, color.strip('rgb()').split(','))) for color in rgb_colors]
    return rgb_tuples


def get_color_recommendations(color_name):
    color_dict = {}

    complementary_color = get_color_scheme_complementary(color_name)
    color_dict["complementary_color"] = complementary_color

    monochromatic_colors = get_color_scheme_monochromatic(color_name)
    color_dict["monochromatic_colors"] = monochromatic_colors

    analogous_colors = get_color_scheme_analogous(color_name)
    color_dict["analogous_colors"] = analogous_colors
    return color_dict


if __name__ == "__main__":
    get_color_recommendations("Brown")

