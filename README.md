# Indoor Localisation

This repository contains the code, notebooks, and generated results for the basic tasks and the final advanced localisation solution.

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

Place the competition dataset under `dataset/data` using this structure:

```text
indoor_localization/
|-- dataset/
|   `-- data/
|       |-- site1/
|       `-- site2/
|-- requirements.txt
`-- ...
```

Run all scripts and notebooks from the repository root. Start the notebooks with:

```bash
jupyter lab
```

The notebooks contain some old package-installation and Google Colab setup cells. These cells can be skipped when running locally.

## Basic tasks

### Waypoint visualisation

Plain coordinate visualisation:

```bash
python waypoint_viz_plt.py
```

Floor-plan overlay:

```bash
python waypoint_visualization.py
```

The generated figures are stored under `waypoint_viz/` and `waypoint_viz_plt/` by site and floor.

### Magnetic-field visualisation

Open `Magnetic_visualization.ipynb` and set the required site and floor in its configuration cell:

```python
floor_data_dir = "./dataset/data/site1/F1"
```

Skip the installation and Google Drive cells, then run the remaining cells in order.

### iBeacon RSS visualisation

Open `iBeacon Viz/iBeacon RSS Visualization.ipynb` and set:

```python
SITE = "site1"
FLOOR = "F4"
DATA_DIR = "./dataset"
```

Update the later floor-information and floor-image paths to use the same site, floor, and data directory. Run the remaining cells in order. Example outputs are stored in `iBeacon Viz/`.

### Wi-Fi RSSI visualisation

Set the dataset, site, and floors near the top of `wifi_viz/wifi_visualization.py`:

```python
filedir = "dataset/data"
siteno = "2"
floornos = ["F1"]
```

Run:

```bash
python wifi_viz/wifi_visualization.py
```

The generated heatmaps are stored under `wifi_viz/output/`.

## Advanced task

### Magnetic and IMU Viterbi localisation

Open `Mag+IMU_viterbi_localization.ipynb` and set:

```python
floor_data_dir = "./dataset/data/site1/F1"
```

Skip the installation and Google Drive cells, then run the remaining cells in order. The final cells print the mean and median positioning errors. This notebook contains the final selected approach for the advanced task.

---

*Experimental files:* `Advanced Task 1_MLP/MLP_iBeacon_vs_Multimodal.ipynb` compares an iBeacon-only MLP with an MLP using iBeacon, Wi-Fi, and geomagnetic features. `pykalman_waypoint_estimation.py` estimates movement from accelerometer and rotation-vector readings using a Kalman filter, with its results stored under `sensor_fusion_viz/`. These approaches were tested but were not selected as the final solution.
