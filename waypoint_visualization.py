from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
import json
from dataset.io_f import read_data_file
data_dir = Path("dataset/data")

for image_path in data_dir.glob("site*/**/*.png"):
    img = Image.open(image_path).convert("RGB")
    img_width_in_px, img_height_in_px = img.size
    site = image_path.parent.parent.name
    floor_info_path = image_path.parent / "floor_info.json"
    with open(floor_info_path, encoding="utf-8") as file:
        map_info = json.load(file)["map_info"]
        floor_height = map_info["height"]
        floor_width = map_info["width"]

    sensor_info_path = image_path.parent / "path_data_files"
    path_points = []
    for sensor_path in sensor_info_path.glob("*.txt"):
        sensor_data = read_data_file(sensor_path)
        for timestamp, x, y in sensor_data.waypoint:
            x_px = round(x / floor_width * img_width_in_px)
            y_px = round(img_height_in_px - (y / floor_height * img_height_in_px))
            path_points.append((x_px, y_px))

    draw = ImageDraw.Draw(img)
    draw.line(path_points, fill=(255, 0, 0), width=2)
    for x, y in path_points:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 100, 255), outline =(255, 255, 255), width=1)

    save_path = Path("waypoint_viz") / site / f"{image_path.parent.name}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)
