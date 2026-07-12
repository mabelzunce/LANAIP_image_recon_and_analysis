import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.widgets import Slider
import os
import re

def read_albira_hdr(hdr_path):
    info = {}

    with open(hdr_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            key = parts[0].lower()

            if key in [
                "x_dimension", "y_dimension", "z_dimension",
                "pixel_size_x", "pixel_size_y", "pixel_size_z",
                "iterations", "data_type", "data_order"
            ]:
                info[key] = float(parts[1]) if "." in parts[1] else int(parts[1])

    shape = (
        info["z_dimension"],
        info["y_dimension"],
        info["x_dimension"]
    )

    voxel_size = (
        info["pixel_size_z"],
        info["pixel_size_y"],
        info["pixel_size_x"]
    )

    iterations = info.get("iterations", None)

    return shape, voxel_size, iterations


def read_albira_img(img_path, hdr_path):
    shape, voxel_size, iterations = read_albira_hdr(hdr_path)

    data = np.fromfile(img_path, dtype="<f4")  # little-endian float32

    if data.size != np.prod(shape):
        raise ValueError(
            f"Size mismatch: expected {np.prod(shape)}, got {data.size}"
        )

    volume = data.reshape(shape)  # (z, y, x)

    return volume, voxel_size, iterations

def read_interfile_hv(hv_path):
    info = {}

    with open(hv_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Must contain a key-value separator
            if ":=" not in line:
                continue

            # Remove leading '!' but keep the line
            if line.startswith("!"):
                line = line[1:].strip()

            key, value = [x.strip().lower() for x in line.split(":=", 1)]
            info[key] = value

    # --- Dimensions ---
    x = int(info["matrix size [1]"])
    y = int(info["matrix size [2]"])
    z = int(info["matrix size [3]"])
    shape = (z, y, x)

    # --- Voxel size (mm) ---
    vx = float(info["scaling factor (mm/pixel) [1]"])
    vy = float(info["scaling factor (mm/pixel) [2]"])
    vz = float(info["scaling factor (mm/pixel) [3]"])
    voxel_size = (vz, vy, vx)

    # --- Data type ---
    if info.get("number format") != "float":
        raise ValueError("Only float data supported")

    if int(info.get("number of bytes per pixel", 0)) != 4:
        raise ValueError("Only 4-byte float supported")

    # --- Byte order ---
    byte_order = info.get("imagedata byte order", "littleendian")
    dtype = "<f4" if byte_order == "littleendian" else ">f4"

    data_file = info.get("name of data file")

    return shape, voxel_size, dtype, data_file

def read_interfile_v(v_path, hv_path):
    shape, voxel_size, dtype, _ = read_interfile_hv(hv_path)

    data = np.fromfile(v_path, dtype=dtype)

    expected = np.prod(shape)
    if data.size != expected:
        raise ValueError(
            f"Size mismatch: expected {expected}, got {data.size}"
        )

    volume = data.reshape(shape)  # (z, y, x)
    name, _ = os.path.splitext(hv_path)

    match = re.search(r"_([0-9]+)$", name)
    iterations = int(match.group(1))
    return volume, voxel_size,iterations
    
def spherical_roi_mask(shape, voxel_size, center_mm, radius_mm):
    dz, dy, dx = voxel_size
    nz, ny, nx = shape

    z = (np.arange(nz) - (nz - 1) / 2) * dz
    y = (np.arange(ny) - (ny - 1) / 2) * dy
    x = (np.arange(nx) - (nx - 1) / 2) * dx

    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    dist2 = (
        (Z - center_mm[0])**2 +
        (Y - center_mm[1])**2 +
        (X - center_mm[2])**2
    )

    return dist2 <= radius_mm**2



def show_overlay(volume, roi_mask, z=None):
    if z is None:
        z = volume.shape[0] // 2

    plt.figure(figsize=(6,6))
    plt.imshow(volume[z], cmap="gray")
    plt.contour(roi_mask[z], colors="r", linewidths=1)
    plt.title(f"Axial slice z={z}")
    plt.axis("off")
    plt.show()

def roi_stats(volume, roi_mask):
    v = volume[roi_mask]
    return {
        "mean":   v.mean(),
        "median": np.median(v),
        "std":    v.std(),
        "min":    v.min(),
        "max":    v.max()
    }

def show_slicer_color(volume, roi_mask, vmin=None, vmax=None):
    nz = volume.shape[0]

    # Find ROI extent
    z_slices = np.where(roi_mask.any(axis=(1, 2)))[0]
    z0 = int(z_slices[len(z_slices)//2]) if len(z_slices) > 0 else nz // 2

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.18)

    base = ax.imshow(volume[z0], cmap="gray", vmin=vmin, vmax=vmax)

    roi_overlay = ax.imshow(
        np.zeros((*roi_mask[z0].shape, 4)),
        interpolation="none"
    )

    ax.set_title(f"Axial slice z={z0}")
    ax.axis("off")

    def update_overlay(z):
        overlay = np.zeros((*roi_mask[z].shape, 4))
        overlay[..., 0] = 1.0              # red
        overlay[..., 3] = roi_mask[z] * 0.4
        roi_overlay.set_data(overlay)

    update_overlay(z0)

    ax_slider = plt.axes([0.2, 0.08, 0.6, 0.03])
    slider = Slider(ax_slider, "Slice", 0, nz - 1, valinit=z0, valstep=1)

    def update(val):
        z = int(slider.val)
        base.set_data(volume[z])
        update_overlay(z)
        ax.set_title(f"Axial slice z={z}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

img_files_b = sorted(glob.glob("IQ/bruker/*.img"))
img_files_s = sorted(glob.glob("IQ/stir/*.v"))
stats_s = {k: [] for k in ["mean","median","std","min","max"]}
stats_fondo_s = {k: [] for k in ["mean","median","std","min","max"]}
stats_b = {k: [] for k in ["mean","median","std","min","max"]}
stats_fondo_b = {k: [] for k in ["mean","median","std","min","max"]}
iters_s = []
iters_b = []

roi_center_b = (-5,6.5,-7)  # bruker tumor
roi_center_fondo_b = (10,4,0)  # bruker fondo
roi_center_fondo_s = (10,-4,0)  # bruker fondo
roi_center_s = (-5,-6.5,7)  # stir tumor
roi_radius_mm = 5.0             # 5 mm sphere
roi_radius = 3.0


for img_path in img_files_b:
    hdr_path = img_path.replace(".img", ".img.hdr")
    
    vol_b, vox_b, it_b = read_albira_img(img_path, hdr_path)
    
    roi_vac_b = spherical_roi_mask(
        vol_b.shape,
        voxel_size=vox_b,
        center_mm=roi_center_b,
        radius_mm=roi_radius
    )
    roi_fon_b = spherical_roi_mask(
        vol_b.shape,
        voxel_size=vox_b,
        center_mm=roi_center_fondo_b,
        radius_mm=roi_radius_mm
    )
    #show_slicer_color(vol_b,roi_vac_b)
    s_tum_b = roi_stats(vol_b, roi_vac_b)
    s_fon_b = roi_stats(vol_b, roi_fon_b)
    for k in stats_b:
        stats_b[k].append(s_tum_b[k])
        stats_fondo_b[k].append(s_fon_b[k])
        
    iters_b.append(it_b)


for img_path in img_files_s:
    hdr_path = img_path.replace(".v", ".hv")
    
    vol_s, vox_s, it_s = read_interfile_v(img_path, hdr_path)
    
    roi_vac_s = spherical_roi_mask(
        vol_s.shape,
        voxel_size=vox_s,
        center_mm=roi_center_s,
        radius_mm=roi_radius
    )
    roi_fon_s = spherical_roi_mask(
        vol_s.shape,
        voxel_size=vox_s,
        center_mm=roi_center_fondo_s,
        radius_mm=roi_radius_mm
    )
    #show_slicer_color(vol_s,roi_vac_s)
    s_vac_s = roi_stats(vol_s, roi_vac_s)
    s_fon_s = roi_stats(vol_s, roi_fon_s)
    for k in stats_s:
        stats_s[k].append(s_vac_s[k])
        stats_fondo_s[k].append(s_fon_s[k])
        
    iters_s.append(it_s)

bruker = np.array(stats_b["mean"])/np.array(stats_fondo_b["mean"])
stir = np.array(stats_s["mean"])/np.array(stats_fondo_s["mean"])
noise_b = np.array(stats_fondo_b["std"])/np.array(stats_fondo_b["mean"])#np.array(stats_fondo_b["std"])/np.array(stats_b["std"])
noise_s = np.array(stats_fondo_s["std"])/np.array(stats_fondo_s["mean"])#np.array(stats_fondo_s["std"])/np.array(stats_s["std"])
plt.figure(figsize=(8,5))

plt.plot(iters_b, noise_b, "o", label="bruker")
plt.plot(iters_s, noise_s, "o", label="stir")

plt.xlabel("Iteraciones")
plt.ylabel("STD fondo")
plt.title("Comparación")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.plot(iters_b, bruker, "o", label="bruker")
plt.plot(iters_s, stir, "o", label="stir")

plt.xlabel("Iteraciones")
plt.ylabel("SOR")
plt.title("Comparación")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
