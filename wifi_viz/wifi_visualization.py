import os
from pathlib import Path
import numpy as np
import pandas as pd
from compute_f import compute_step_positions, split_ts_seq
import matplotlib.pyplot as plt
from matplotlib.image import imread
import json
import sys

filedir = 'indoor-location-competition-20/data'
siteno = '1'
floornos = ['B1','F2','F4']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/output'

def get_output_image_path(siteno, floorno, bssid):
    heatmap_name = f'wifiheatmap_s{siteno}_{floorno}_{bssid}'
    return os.path.join(SCRIPT_DIR, f'{heatmap_name}.png')

def extract_wifi_data(filedir, siteno, floorno):
    mwi_datas = {}
    path = filedir+'/site'+siteno+'/'+floorno+'/'+'path_data_files'+'/'
    path_filenames = list(Path(path).resolve().glob("*.txt"))
    
    for path_filename in path_filenames:

        with open(path_filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        wifi = []
        acce = []
        ahrs = []
        waypoint = []

        for line_data in lines:
            line_data = line_data.strip()
            if not line_data or line_data[0] == '#':
                continue

            line_data = line_data.split('\t')

            if line_data[1] == 'TYPE_WIFI':
                # print(line_data)
                # ['1574572277475', 'TYPE_WIFI', 'laomiaozhubao', '74:59:09:e1:3e:dc', '-48', '2437', '1574572275912']
                sys_ts = line_data[0]
                ssid = line_data[2]
                bssid = line_data[3]
                rssi = line_data[4]
                # 5 is frequency
                lastseen_ts = line_data[6]
                wifi_data = [sys_ts, ssid, bssid, rssi, lastseen_ts]
                wifi.append(wifi_data)
                continue

            if line_data[1] == 'TYPE_ACCELEROMETER':
                acce.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
                continue

            if line_data[1] == 'TYPE_ROTATION_VECTOR':
                ahrs.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
                continue

            if line_data[1] == 'TYPE_WAYPOINT':
                waypoint.append([int(line_data[0]), float(line_data[2]), float(line_data[3])])

        acce = np.array(acce)
        ahrs = np.array(ahrs)
        waypoint = np.array(waypoint)
        wifi = np.array(wifi)

        acce_datas = acce
        ahrs_datas = ahrs
        wifi_datas = wifi
        posi_datas = waypoint

        step_positions = compute_step_positions(acce, ahrs, waypoint)

        if wifi_datas.size != 0:
            sep_tss = np.unique(wifi_datas[:, 0].astype(float))
            wifi_datas_list = split_ts_seq(wifi_datas, sep_tss)
            for wifi_ds in wifi_datas_list:
                diff = np.abs(step_positions[:, 0] - float(wifi_ds[0, 0]))
                index = np.argmin(diff)
                target_xy_key = tuple(step_positions[index, 1:3])
                if target_xy_key in mwi_datas:
                    mwi_datas[target_xy_key]['wifi'] = np.append(mwi_datas[target_xy_key]['wifi'], wifi_ds, axis=0)
                else:
                    mwi_datas[target_xy_key] = {
                        'magnetic': np.zeros((0, 4)),
                        'wifi': wifi_ds,
                        'ibeacon': np.zeros((0, 3))
                    }
    return mwi_datas


def build_wifi_rssi(mwi_datas):
    wifi_rssi = {}
    for position_key in mwi_datas:
        wifi_data = mwi_datas[position_key]['wifi']
        for wifi_d in wifi_data:
            bssid = wifi_d[2]
            rssi = int(wifi_d[3])

            if bssid in wifi_rssi:
                position_rssi = wifi_rssi[bssid]
                if position_key in position_rssi:
                    old_rssi = position_rssi[position_key][0]
                    old_count = position_rssi[position_key][1]
                    position_rssi[position_key][0] = (old_rssi * old_count + rssi) / (old_count + 1)
                    position_rssi[position_key][1] = old_count + 1
                else:
                    position_rssi[position_key] = np.array([rssi, 1])
            else:
                position_rssi = {}
                position_rssi[position_key] = np.array([rssi, 1])

            wifi_rssi[bssid] = position_rssi
    return wifi_rssi


def list_ap_for_wifi(wifi_rssi, count):
    n_wifi_bssids = list(wifi_rssi.keys())[0:count]
    print('3 Unique APs for which heatmap is generated:\n')
    n_wifi_bssids = np.unique(n_wifi_bssids)[:3]
    for bssid in n_wifi_bssids:
        print(bssid)
    return n_wifi_bssids


def visualize_wifi_data(wifi_rssi, bssid):
    with open(f"{filedir}/site{siteno}/{floorno}/floor_info.json", 'r') as file:
        floor_data = json.load(file)

    heat_positions = np.array(list(wifi_rssi[bssid].keys()))
    heat_values = np.array(list(wifi_rssi[bssid].values()))[:, 0]

    width = floor_data['map_info']['width']
    height = floor_data['map_info']['height']

    px, py = heat_positions[:, 0], heat_positions[:, 1]
    rssi = np.asarray(heat_values, dtype=float)

    floor_img = imread(f"{filedir}/site{siteno}/{floorno}/floor_image.png")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Base layer: floor plan
    ax.imshow(floor_img, extent=[0, width, 0, height], origin="upper", zorder=0)

    sc = ax.scatter(px, py, c=rssi, cmap="jet", s=45, edgecolors="face", linewidths=0.6,
                     vmin=rssi.min(), vmax=rssi.max(), zorder=2, label="measurement points")

    cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("RSSI (dBm) — higher = stronger signal")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"WiFi RSSI heatmap — BSSID {bssid}")
    ax.legend(loc="upper right")

    ax.set_xlim(max(0, px.min() - 35), min(width, px.max() + 35))
    ax.set_ylim(max(0, py.min() - 35), min(height, py.max() + 35))

    output_image_path = get_output_image_path(siteno, floorno, bssid)
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path)
    print(f"Saved heatmap to {output_image_path}")
    plt.close(fig)


if __name__ == '__main__':
    # mwi_datas = extract_wifi_data(filedir, siteno, floorno)
    # wifi_rssi = build_wifi_rssi(mwi_datas)
    # list_ap_for_wifi(wifi_rssi, 10)

    # bssid = list(wifi_rssi.keys())[0]
    # visualize_wifi_data(wifi_rssi, bssid)

    for floorno in floornos:
        print(f"\nProcessing Site {siteno}, Floor {floorno}")
 
        mwi_datas = extract_wifi_data(filedir, siteno, floorno)
 
        if not mwi_datas:
            print(f" [warn] no wifi/position data extracted for floor {floorno}, skipping.")
            continue
 
        wifi_rssi = build_wifi_rssi(mwi_datas)
 
        if not wifi_rssi:
            print(f"no APs found for floor {floorno}, skipping.")
            continue
 
        aps = list_ap_for_wifi(wifi_rssi, 10)

        for ap in aps:
            bssid = ap
            visualize_wifi_data(wifi_rssi, bssid)
