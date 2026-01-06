###############################################################################
# Spatial Data, GIS, and Remote Sensing
# Panoptic Segmentation (Toy Demo) on a Free Remote-Sensing Raster (Python)
# Author: Jared Edgerton
# Date: date.today()
#
# Coding lab focus:
#   1) GeoPandas workflows: vector data, CRS, reprojection, spatial joins
#   2) Raster workflows: rasterio reading, windowing, rasterization
#   3) Mapping: plotting rasters + vectors together
#   4) Neural network demo: semantic segmentation + simple instance extraction
#      (a "panoptic-style" output: stuff classes + thing instances)
#
# Application paper: Harari, M. (2020) AER
# Application paper: Panagopoulos et al. (2023) Land
# Pre-class video: Spatial data formats + coordinate systems
#
# Teaching note (important):
# - This file is intentionally written as a "hard-coded" sequential workflow.
# - No user-defined functions.
# - No conditional statements (no if/else).
# - Steps are repeated explicitly so students can follow and modify each piece.
#
# Important realism note:
# - True panoptic segmentation usually requires labeled instance + semantic data.
# - Here we create toy vector labels (buildings=instances; roads=stuff) and
#   rasterize them to train a small segmentation model on one patch.
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# If you do not have these installed, run (in Terminal / Anaconda Prompt):
#   pip install numpy pandas matplotlib rasterio geopandas shapely pyproj scipy torch requests
#
# Notes:
# - On Windows, GeoPandas is easiest via conda-forge.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import requests

import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from rasterio.features import rasterize

import geopandas as gpd
from shapely.geometry import box, LineString, Point

import scipy.ndimage as ndi

import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import date

np.random.seed(123)
torch.manual_seed(123)

# -----------------------------------------------------------------------------
# Part 1: Download + Read a Free Georeferenced Raster (Warm-up)
# -----------------------------------------------------------------------------
# We want a file that is:
# - free
# - small enough for class
# - georeferenced (has CRS + transform)

raster_url = "https://github.com/rasterio/rasterio/raw/main/tests/data/RGB.byte.tif"
raster_path = "RGB.byte.tif"

r = requests.get(raster_url)
open(raster_path, "wb").write(r.content)

src = rasterio.open(raster_path)

print("\n------------------------------")
print("Raster metadata")
print("------------------------------")
print("Path:", raster_path)
print("Driver:", src.driver)
print("CRS:", src.crs)
print("Bounds:", src.bounds)
print("Width, Height:", src.width, src.height)
print("Band count:", src.count)
print("Dtype:", src.dtypes)
print("Transform:", src.transform)

img_full = src.read().astype(np.float32)  # (bands, rows, cols)
img_full_disp = img_full / np.maximum(img_full.max(), 1.0)

plt.figure()
plt.imshow(np.transpose(img_full_disp, (1, 2, 0)))
plt.title("Remote-sensing raster (RGB.byte.tif) — full image")
plt.axis("off")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# Part 2: Raster Windowing (Work on a Small Patch)
# -----------------------------------------------------------------------------
# Extract a 256x256 window so training is fast.

win_col_off = 100
win_row_off = 100
win_width   = 256
win_height  = 256

win = Window(win_col_off, win_row_off, win_width, win_height)

img = src.read(window=win).astype(np.float32)        # (bands, H, W)
win_tfm = window_transform(win, src.transform)       # affine transform for patch
img_disp = img / np.maximum(img.max(), 1.0)

plt.figure()
plt.imshow(np.transpose(img_disp, (1, 2, 0)))
plt.title("256x256 raster patch (windowed read)")
plt.axis("off")
plt.tight_layout()
plt.show()

C, H, W = img.shape
print("\nPatch shape (C, H, W):", img.shape)

# -----------------------------------------------------------------------------
# Part 3: Vector Data in a CRS (GeoPandas + Shapely)
# -----------------------------------------------------------------------------
# Create toy "building polygons" (instances) and "roads" (stuff) in the raster CRS.

x0, y0 = (win_tfm * (0, 0))
x1, y1 = (win_tfm * (W, H))

xmin = min(x0, x1)
xmax = max(x0, x1)
ymin = min(y0, y1)
ymax = max(y0, y1)

print("\nPatch bounds (approx):")
print("xmin, xmax:", xmin, xmax)
print("ymin, ymax:", ymin, ymax)

# -----------------------------------------------------------------------------
# Step 1: Building polygons (instances)
# -----------------------------------------------------------------------------
num_buildings = 40

bx = np.random.uniform(xmin, xmax, size=num_buildings)
by = np.random.uniform(ymin, ymax, size=num_buildings)

bwidth  = np.random.uniform((xmax - xmin) * 0.01, (xmax - xmin) * 0.05, size=num_buildings)
bheight = np.random.uniform((ymax - ymin) * 0.01, (ymax - ymin) * 0.05, size=num_buildings)

building_geoms = []
building_ids = []

for i in range(num_buildings):
    building_geoms.append(box(bx[i], by[i], bx[i] + bwidth[i], by[i] + bheight[i]))
    building_ids.append(i + 1)

buildings_gdf = gpd.GeoDataFrame(
    {"building_id": building_ids, "class_name": ["building"] * num_buildings},
    geometry=building_geoms,
    crs=src.crs
)

print("\nBuildings GeoDataFrame (head):")
print(buildings_gdf.head())

# -----------------------------------------------------------------------------
# Step 2: Roads as buffered lines (stuff)
# -----------------------------------------------------------------------------
road_lines = []
road_ids = []

road_lines.append(LineString([(xmin, (ymin + ymax) / 2), (xmax, (ymin + ymax) / 2)]))
road_ids.append(1)

road_lines.append(LineString([((xmin + xmax) / 2, ymin), ((xmin + xmax) / 2, ymax)]))
road_ids.append(2)

road_lines.append(LineString([(xmin, ymin), (xmax, ymax)]))
road_ids.append(3)

roads_buffer_dist = (xmax - xmin) * 0.01
road_polys = [ln.buffer(roads_buffer_dist) for ln in road_lines]

roads_gdf = gpd.GeoDataFrame(
    {"road_id": road_ids, "class_name": ["road"] * len(road_polys)},
    geometry=road_polys,
    crs=src.crs
)

print("\nRoads GeoDataFrame (head):")
print(roads_gdf.head())

# -----------------------------------------------------------------------------
# Step 3: CRS transformation example (projected CRS -> WGS84)
# -----------------------------------------------------------------------------
buildings_wgs84 = buildings_gdf.to_crs("EPSG:4326")
roads_wgs84 = roads_gdf.to_crs("EPSG:4326")

print("\nBuildings in WGS84 (head):")
print(buildings_wgs84.head())

# -----------------------------------------------------------------------------
# Step 4: Spatial join example (points -> buildings)
# -----------------------------------------------------------------------------
num_points = 300
px = np.random.uniform(xmin, xmax, size=num_points)
py = np.random.uniform(ymin, ymax, size=num_points)

points = gpd.GeoDataFrame(
    {"pt_id": np.arange(1, num_points + 1)},
    geometry=[Point(px[i], py[i]) for i in range(num_points)],
    crs=src.crs
)

points_joined = gpd.sjoin(points, buildings_gdf[["building_id", "geometry"]], how="left", predicate="within")
inside_counts = points_joined["building_id"].notna().sum()
print("\nPoints inside building polygons:", int(inside_counts), "out of", num_points)

# -----------------------------------------------------------------------------
# Part 4: Rasterization (Vectors -> Pixel Masks)
# -----------------------------------------------------------------------------
# Semantic mask:
#   0 = background
#   1 = road (stuff)
#   2 = building (thing)
#
# Instance mask:
#   0 = not building
#   1..K = building instance IDs

road_shapes = [(geom, 1) for geom in roads_gdf.geometry]
roads_mask = rasterize(road_shapes, out_shape=(H, W), transform=win_tfm, fill=0, dtype=np.uint8)

building_sem_shapes  = [(geom, 2) for geom in buildings_gdf.geometry]
building_inst_shapes = [(geom, bid) for geom, bid in zip(buildings_gdf.geometry, buildings_gdf["building_id"])]

buildings_sem_mask = rasterize(building_sem_shapes, out_shape=(H, W), transform=win_tfm, fill=0, dtype=np.uint8)
buildings_inst_mask = rasterize(building_inst_shapes, out_shape=(H, W), transform=win_tfm, fill=0, dtype=np.int32)

semantic_mask = np.zeros((H, W), dtype=np.uint8)
semantic_mask = np.maximum(semantic_mask, roads_mask)
semantic_mask = np.maximum(semantic_mask, buildings_sem_mask)

instance_mask = buildings_inst_mask.copy()

print("\nMask summaries:")
print("Semantic classes present:", np.unique(semantic_mask))
print("Instance IDs present (min/max):", int(instance_mask.min()), int(instance_mask.max()))

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(np.transpose(img_disp, (1, 2, 0))); plt.title("Raster patch"); plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(semantic_mask); plt.title("Semantic mask (0 bg, 1 road, 2 building)"); plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(instance_mask); plt.title("Instance mask (building IDs)"); plt.axis("off")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# Part 5: Neural Network — Semantic Segmentation (PyTorch)
# -----------------------------------------------------------------------------
# Train a small encoder-decoder CNN to predict semantic_mask on the patch.

X = img / np.maximum(img.max(), 1.0)
X_t = torch.tensor(X[None, :, :, :], dtype=torch.float32)
y_t = torch.tensor(semantic_mask[None, :, :], dtype=torch.long)

num_classes = 3

conv1 = nn.Conv2d(C, 16, kernel_size=3, padding=1)
conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
pool1 = nn.MaxPool2d(2)

conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
pool2 = nn.MaxPool2d(2)

up1 = nn.Upsample(scale_factor=2, mode="nearest")
conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
conv6 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

up2 = nn.Upsample(scale_factor=2, mode="nearest")
conv7 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
conv8 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

params = (
    list(conv1.parameters()) + list(conv2.parameters()) +
    list(conv3.parameters()) + list(conv4.parameters()) +
    list(conv5.parameters()) + list(conv6.parameters()) +
    list(conv7.parameters()) + list(conv8.parameters()) +
    list(conv_out.parameters())
)

opt = torch.optim.Adam(params, lr=0.01)

epochs = 60

for epoch in range(1, epochs + 1):
    h = F.relu(conv1(X_t))
    h = F.relu(conv2(h))
    h = pool1(h)

    h = F.relu(conv3(h))
    h = F.relu(conv4(h))
    h = pool2(h)

    h = up1(h)
    h = F.relu(conv5(h))
    h = F.relu(conv6(h))

    h = up2(h)
    h = F.relu(conv7(h))
    h = F.relu(conv8(h))

    logits = conv_out(h)

    loss = F.cross_entropy(logits, y_t)

    opt.zero_grad()
    loss.backward()
    opt.step()

    pred = torch.argmax(logits, dim=1)
    acc = (pred == y_t).float().mean().item()

    print("Epoch", epoch, "| loss:", float(loss.detach().cpu().numpy()), "| pixel_acc:", acc)

# -----------------------------------------------------------------------------
# Part 6: Panoptic-Style Output (Semantic + Instances)
# -----------------------------------------------------------------------------
# Convert predicted building pixels into instances via connected components.

pred_sem = pred.detach().cpu().numpy()[0].astype(np.uint8)
pred_building_mask = (pred_sem == 2).astype(np.uint8)

pred_instances, pred_num_instances = ndi.label(pred_building_mask)

print("\nPanoptic post-processing:")
print("  Predicted building instances:", int(pred_num_instances))

# Panoptic ID encoding (teaching convenience):
# - road pixels: 1000 (class 1, instance 0)
# - building pixels: 2000 + instance_id (class 2, instance >=1)
panoptic = np.zeros((H, W), dtype=np.int32)
panoptic = np.where(pred_sem == 1, 1000, panoptic)
panoptic = np.where(pred_sem == 2, 2000 + pred_instances, panoptic)

# -----------------------------------------------------------------------------
# Part 7: Visualization (Raster + Semantic + Panoptic)
# -----------------------------------------------------------------------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(np.transpose(img_disp, (1, 2, 0))); plt.title("Raster patch"); plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(pred_sem); plt.title("Predicted semantic classes"); plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(panoptic); plt.title("Panoptic-style IDs\n(road=1000, building=2000+id)"); plt.axis("off")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# Part 8: Simple Diagnostics (Pixel Accuracy vs Toy Labels)
# -----------------------------------------------------------------------------
pixel_acc = (pred_sem == semantic_mask).mean()
print("\nPixel accuracy vs toy labels:", float(pixel_acc))
