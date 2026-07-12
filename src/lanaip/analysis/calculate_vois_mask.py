
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.widgets import Slider
from skimage import measure
import pandas as pd
from PIL import Image


mask_path = "Phantom/IQ_calc/mask_adq2.nii"
pet_path  = "Phantom/IQ_calc/adq2_kbq_cc.nii"


def make_label_colormap(labels):
    """
    Build an RGBA colormap and a BoundaryNorm so each integer label maps to one color.
    label 0 will be transparent.
    """
    # choose colors for non-zero labels (extend as needed)
    base_colors = [
        (1.0, 0.0, 0.0, 1.0),  # red
        (0.0, 1.0, 0.0, 1.0),  # green
        (0.0, 0.0, 1.0, 1.0),  # blue
        (1.0, 1.0, 0.0, 1.0),  # yellow
        (1.0, 0.0, 1.0, 1.0),  # magenta
        (0.0, 1.0, 1.0, 1.0),  # cyan
    ]
    labels = list(labels)
    labels_sorted = sorted(labels)
    # ensure 0 included
    if 0 not in labels_sorted:
        labels_sorted.insert(0, 0)
    n = len(labels_sorted)

    colors = []
    for lab in labels_sorted:
        if lab == 0:
            colors.append((0,0,0,0.0))  # transparent for background
        else:
            colors.append(base_colors[(lab-1) % len(base_colors)])
    cmap = ListedColormap(colors)
    # boundaries midpoints between labels
    boundaries = [lab - 0.5 for lab in labels_sorted] + [labels_sorted[-1] + 0.5]
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm, labels_sorted

def build_rgb_overlay(pet_slice, mask_slice, cmap, norm, alpha=0.35):
    """Return an RGB image blending pet grayscale and colored mask (mask applied with alpha)."""
    # Normalize pet to 0..1 for display
    pmin, pmax = np.percentile(pet_slice, 2), np.percentile(pet_slice, 98)
    pet_norm = np.clip((pet_slice - pmin) / (pmax - pmin + 1e-12), 0, 1)

    # base grayscale rgb
    rgb = np.stack([pet_norm, pet_norm, pet_norm], axis=-1)

    # colorize mask using cmap+norm
    mask_rgba = cmap(norm(mask_slice))
    mask_rgb = mask_rgba[..., :3]
    mask_alpha = mask_rgba[..., 3]
    # but we want the same alpha for all nonzero labels (preserve transparency 0 for label 0)
    mask_alpha = (mask_slice != 0).astype(float) * alpha

    # blend: out = (1 - a)*rgb + a*mask_rgb
    out = (1.0 - mask_alpha[..., None]) * rgb + (mask_alpha[..., None]) * mask_rgb
    return out

def triple_slider_viewer(pet, mask):

    # Ensure axes order is (z,y,x) — adjust if your arrays are (x,y,z)
    if pet.shape != mask.shape:
        raise ValueError("pet and mask must have same shape")

    nz = pet.shape[0]
    init_slice = nz // 2

    unique_labels = np.unique(mask)
    cmap, norm, labels_sorted = make_label_colormap(unique_labels)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax_pet, ax_mask, ax_fusion = axes

    # initial data
    pet_slice = pet[init_slice, :, :]
    mask_slice = mask[init_slice, :, :]

    # left: pet grayscale
    im_pet = ax_pet.imshow(pet_slice, cmap="gray")
    ax_pet.set_title("PET")
    ax_pet.axis("off")

    # center: mask discrete labels with norm
    im_mask = ax_mask.imshow(mask_slice, cmap=cmap, vmin=0, vmax=labels_sorted[-1])
    ax_mask.set_title("Mask (labels)")
    ax_mask.axis("off")

    # right: fusion RGB
    fusion = build_rgb_overlay(pet_slice, mask_slice, cmap, norm, alpha=0.35)
    im_fusion = ax_fusion.imshow(fusion)
    ax_fusion.set_title("Fusion")
    ax_fusion.axis("off")

    # Add contours to fusion to emphasize borders
    contours = {}
    for lab in labels_sorted:
        if lab == 0: 
            continue
        cs = measure.find_contours((mask_slice == lab).astype(np.uint8), 0.5)
        contours[lab] = cs
        # draw initial contours
        for contour in cs:
            ax_fusion.plot(contour[:, 1], contour[:, 0], linewidth=1.0, color='white', alpha=0.9)

    fig.suptitle(f"Slice {init_slice}/{nz-1}", fontsize=14)
    plt.subplots_adjust(bottom=0.12)

    # Slider
    ax_slider = plt.axes([0.2, 0.04, 0.6, 0.03])
    slider = Slider(ax_slider, 'slice', 0, nz-1, valinit=init_slice, valstep=1)

    def update(val):
        s = int(slider.val)
        pet_slice = pet[s, :, :]
        mask_slice = mask[s, :, :]

        # update pet
        pmin, pmax = np.percentile(pet_slice, 2), np.percentile(pet_slice, 98)
        im_pet.set_data(pet_slice)
        #im_pet.set_clim(vmin=pmin, vmax=pmax)

        # update mask (discrete)
        im_mask.set_data(mask_slice)

        # update fusion
        fusion = build_rgb_overlay(pet_slice, mask_slice, cmap, norm, alpha=0.35)
        im_fusion.set_data(fusion)

        # remove previous contours and add new ones
        for line in ax_fusion.lines[:]:
            line.remove()
        for lab in labels_sorted:
            if lab == 0:
                continue
            cs = measure.find_contours((mask_slice == lab).astype(np.uint8), 0.5)
            for contour in cs:
                ax_fusion.plot(contour[:, 1], contour[:, 0], linewidth=1.0, color='white', alpha=0.9)

        fig.suptitle(f"Slice {s}/{nz-1}", fontsize=14)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
        
pet  = Image.open(pet_path)
mask = Image.open(mask_path)


# --- METRICS ---
vois = np.unique(mask)
vois = vois[vois != 0]

res = []
for v in vois:
    vals = pet[mask == v]
    res.append({
        "VOI": v,
        "Mean": float(vals.mean()),
        "Std":  float(vals.std()),
        "Min":  float(vals.min()),
        "Max":  float(vals.max()),
    })
df = pd.DataFrame(res)
print(df)

triple_slider_viewer(pet, mask)
