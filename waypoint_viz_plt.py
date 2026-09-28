from PIL import Image
import numpy as np
from pathlib import Path
import json
from dataset.io_f import read_data_file
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator 
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
    plt.figure(figsize=(10, 10*floor_height/floor_width))
    colored_points = None
    for sensor_path in sensor_info_path.glob("*.txt"):
        sensor_data = read_data_file(sensor_path)
        waypoints = sensor_data.waypoint
        print(len(waypoints))
        x = waypoints[:,1]
        y = waypoints[:,2]
        progress = np.linspace(0, 1, len(x))
        plt.plot(x, y, color='red', linewidth=1)
        colored_points = plt.scatter(x,y,c=progress,cmap="viridis",vmin=0,vmax=1,s=10)

    plt.colorbar(colored_points, label='Progress')
    plt.xlim(0, floor_width)
    plt.ylim(0, floor_height)
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    floor = image_path.parent.name
    plt.title(f'Waypoints for {floor} in {site}')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().xaxis.set_major_locator(MultipleLocator(20))
    plt.gca().yaxis.set_major_locator(MultipleLocator(20))
    plt.grid(True, linestyle='--', linewidth=0.5)
    save_path = Path("waypoint_viz") / site / f"{image_path.parent.name}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
          save_path,
          dpi=200,
          bbox_inches="tight",
          facecolor="white",
      )

    plt.close()
