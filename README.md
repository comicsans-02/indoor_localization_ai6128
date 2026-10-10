# Indoor Localisation

This repository contains the code, notebooks, and generated results for the basic and advanced tasks.

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

## Advanced tasks

### iBeacon-only and multimodal MLP

Open `Advanced Task 1_MLP/MLP_iBeacon_vs_Multimodal.ipynb` and set:

```python
DATA_DIR = "./dataset"
```

Skip the package-installation cell and run the notebook in order. The result figures are stored in `Advanced Task 1_MLP/`.

### Magnetic and IMU Viterbi localisation

Open `Mag+IMU_viterbi_localization.ipynb` and set:

```python
floor_data_dir = "./dataset/data/site1/F1"
```

Skip the installation and Google Drive cells, then run the remaining cells in order. The final cells print the mean and median positioning errors.

### Accelerometer-based Kalman filter

Run the default evaluation for `site1/B1`, `site1/F2`, and `site1/F4`:

```bash
python pykalman_waypoint_estimation.py
```

Different floors can be selected using command-line arguments:

```bash
python pykalman_waypoint_estimation.py --site site1 --floors B1 F1 F2
```

The plots and result summary are stored under `sensor_fusion_viz/`.
