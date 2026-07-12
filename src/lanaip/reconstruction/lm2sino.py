#!/usr/bin/env python3
"""
Qué hace:
 - Parsea el .lm del Albira: lee header y de ahí se define el modo de leer los eventos (v1 filever<6/v2 filever>=6)
 - Rebinea desde 300x300
 - Mapea las posiciones XY a las coordenadas físicas en la proyección del anillo
 - Construye sinograma (bins_axial, bins_views, tangential). Forzado tamaño de 
 - Escribe los archivos compatibles con STIR: .s (binario) y .hs (header)
 - Escribe los archivos compatibles con MRIcro (.hdr/.img)
 - Plotea un mapa de los módilos/eventos detectados (hitmap). Usado para ver que lee bien los eventos y los posiciona
 - Progress bar porque a vecer tarda un poco y no hay paciencia
 - Hay un grafico para pasar los segmentos del sinograma (colorbar en log para no tener que poner algo con colores)
"""

import os
import struct
import math
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
# ---------------------------
# LM parser (header + events)
# ---------------------------
header_size = 176
singles_flag = 0x100

fmt_v1 = "<H2xHHfHHf4xdf4x"   # 40 bytes
size_v1 = struct.calcsize(fmt_v1)
fmt_v2 = "<dfffHHHHHH"        # 32 bytes
size_v2 = struct.calcsize(fmt_v2)

class lm_header:
    def __init__(self, raw: bytes):
        if len(raw) < header_size:
            raise ValueError("Header too short")
        self.identifier = raw[0:16].decode("latin1", errors="replace").rstrip("\x00")
        self.rawCounts = struct.unpack("<d", raw[16:24])[0]
        self.acqTime = struct.unpack("<d", raw[24:32])[0]
        self.activity = struct.unpack("<d", raw[32:40])[0]
        self.isotope = raw[40:56].decode("latin1", errors="replace").rstrip("\x00")
        self.detectorSizeX = struct.unpack("<d", raw[56:64])[0]
        self.detectorSizeY = struct.unpack("<d", raw[64:72])[0]
        self.startTime = struct.unpack("<d", raw[72:80])[0]
        self.measurementTime = struct.unpack("<d", raw[80:88])[0]
        self.moduleNumber = struct.unpack("<i", raw[88:92])[0]
        self.ringNumber = struct.unpack("<i", raw[92:96])[0]
        self.ringDistance = struct.unpack("<d", raw[96:104])[0]
        self.detectorDistance = struct.unpack("<d", raw[104:112])[0]
        self.isotopeHalfLife = struct.unpack("<d", raw[112:120])[0]
        self.reserved = raw[120:152]
        self.version_bytes = raw[152:154]
        self.reserved2 = raw[154:156]
        self.gatePeriod = struct.unpack("<d", raw[156:164])[0]
        self.reserved3 = raw[164:176]

    def is_v2(self):
        return self.version_bytes[0] > 5

def parse_event_v1(raw: bytes):
    if len(raw) != size_v1:
        raise ValueError(f"v1 event must be {size_v1} bytes (got {len(raw)})")
    unpacked = struct.unpack(fmt_v1, raw)
    pair = unpacked[0]
    e1_x = unpacked[1]; e1_y = unpacked[2]; e1_e = unpacked[3]
    e2_x = unpacked[4]; e2_y = unpacked[5]; e2_e = unpacked[6]
    time = unpacked[7]; amount = unpacked[8]
    return {
        "pair": pair, "time": time, "amount": amount,
        "x1": int(e1_x), "y1": int(e1_y), "e1": e1_e,
        "x2": int(e2_x), "y2": int(e2_y), "e2": e2_e,
        "gate": 0
    }

def parse_event_v2(raw: bytes):
    if len(raw) != size_v2:
        raise ValueError(f"v2 event must be {size_v2} bytes (got {len(raw)})")
    unpacked = struct.unpack(fmt_v2, raw)
    time = unpacked[0]; e1 = unpacked[1]; e2 = unpacked[2]; amount = unpacked[3]
    x1 = int(unpacked[4]); y1 = int(unpacked[5]); x2 = int(unpacked[6]); y2 = int(unpacked[7])
    pair = int(unpacked[8]); gate = int(unpacked[9])
    return {
        "pair": pair, "time": time, "amount": amount,
        "x1": x1, "y1": y1, "e1": e1,
        "x2": x2, "y2": y2, "e2": e2,
        "gate": gate
    }

# ---------------------------
# Geaometría del scanner
# ---------------------------
modules_per_ring = 8
num_rings = 3
total_modules = modules_per_ring * num_rings
crys_module = 300
#REBIN = 10
rebin_y = 6.25
rebin_x = 3
ring_gap = 3
module_bin_x = int(crys_module // rebin_x)
module_bin_y = int(crys_module // rebin_y)
axial_fov = 150
bins_axial = int(2*sum(range(int(num_rings * module_bin_y+(num_rings-1)*ring_gap))) + (num_rings * module_bin_y+(num_rings-1)*ring_gap)) 
segments = 2 * axial_fov - 1 
module_size = 48.0
ring_radius = 117.0/2
crystal_size = module_size / module_bin_x 
transaxial_fov = 90
bins_views =   int(module_size*modules_per_ring/2 +1)
bins_tang = int(round(transaxial_fov/crystal_size)) +1  #90 mm FOV transaxial
energy_resol = 0.17

pair_map = {0:(0,4),1:(1,5),2:(2,6),3:(3,7),4:(0,3),5:(0,5),6:(1,4),7:(1,6),
8:(2,5),9:(2,7),10:(3,6),11:(4,7),12:(8,12),13:(9,13),14:(10,14),
15:(11,15),16:(8,11),17:(8,13),18:(9,12),19:(9,14),20:(10,13),
21:(10,15),22:(11,14),23:(12,15),24:(0,12),25:(4,8),26:(1,13),
27:(5,9),28:(2,14),29:(6,10),30:(3,15),31:(7,11),32:(0,11),
33:(0,13),34:(3,8),35:(5,8),36:(1,12),37:(1,14),38:(4,9),
39:(6,9),40:(2,13),41:(2,15),42:(5,10),43:(7,10),44:(3,14),
45:(6,11),46:(4,15),47:(7,12),48:(16,20),49:(17,21),50:(18,22),
51:(19,23),52:(16,19),53:(16,21),54:(17,20),55:(17,22),56:(18,21),
57:(18,23),58:(19,22),59:(20,23),60:(8,20),61:(12,16),62:(9,21),
63:(13,17),64:(10,22),65:(14,18),66:(11,23),67:(15,19),68:(8,19),
69:(8,21),70:(11,16),71:(13,16),72:(9,20),73:(9,22),74:(12,17),
75:(14,17),76:(10,21),77:(10,23),78:(13,18),79:(15,18),80:(11,22),
81:(14,19),82:(12,23),83:(15,20),84:(0,20),85:(4,16),86:(1,21),
87:(5,17),88:(2,22),89:(6,18),90:(3,23),91:(7,19),92:(0,19),
93:(0,21),94:(3,16),95:(5,16),96:(1,20),97:(1,22),98:(4,17),
99:(6,17),100:(2,21),101:(2,23),102:(5,18),103:(7,18),104:(3,22),
105:(6,19),106:(4,23),107:(7,20)}

# ---------------------------
# Mapeo detector -> coordenadas físicas
# ---------------------------
def module_ring(module_index):
    ring = module_index // modules_per_ring
    mod_in_ring = module_index % modules_per_ring
    return ring, mod_in_ring


def detector_xy_mm(module_index, x_crystal):
    ring, mod = module_ring(module_index)
    local_x_mm =  (module_bin_x / 2.0) - x_crystal
    
    theta_mod = (2/4 + 1/4*mod) * math.pi
    Y_mod = ring_radius * math.cos(theta_mod) 
    X_mod = ring_radius * math.sin(theta_mod) 
    X = X_mod + local_x_mm * crystal_size * math.cos(theta_mod)
    Y = Y_mod - local_x_mm * crystal_size * math.sin(theta_mod)    
    return X, Y, ring

def lor_view(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    
    xm = 0.5 * (x1 + x2)
    ym = 0.5 * (y1 + y2)
    phi = math.atan2(dy, dx) 
    if phi < 0 :
    # View index (angle)
        phi = (math.pi + phi) #+ math.pi/2
        swap = 1 
    else:
        phi = phi # + math.pi / 2
        swap = 0  
    view_mid = (phi)/math.pi * (bins_views-1)
    view_idx_inf = int(np.floor((phi)/math.pi * (bins_views-1)))
    view_idx_sup = int(np.ceil((phi)/math.pi * (bins_views-1)))
    w_view_inf = 1-(view_mid - view_idx_inf)
    w_view_sup = 1-(view_idx_sup - view_mid)        
    #view_bin = int((phi)/math.pi * (bins_views - 1))
    # RADON
    
    s =  xm * math.sin(phi) - ym * math.cos(phi)
    
    crystal_size_cor = transaxial_fov / (bins_tang) 
    tang_mid = (s / crystal_size_cor  + (bins_tang-1) / 2)
    tang_inf = int(np.floor(s / crystal_size_cor + (bins_tang-1) / 2))
    tang_sup = int(np.ceil(s / crystal_size_cor + (bins_tang-1) / 2))
    
    w_tang_inf = 1-(tang_mid - tang_inf)
    w_tang_sup = 1-(tang_sup - tang_mid)
   
    if tang_inf >=0 and tang_sup < bins_tang:
        return view_idx_inf, view_idx_sup, w_view_inf, w_view_sup, tang_inf,tang_sup,w_tang_inf,w_tang_sup, swap

# ---------------------------
# Sinograma y visualizador
# ---------------------------
def build_sinogram_and_hitmap(lm_path, force_v2=False, force_v1=False):
    sino = np.zeros((bins_axial, bins_views, bins_tang), dtype=np.float32)
    hitmap = np.zeros((num_rings * module_bin_y + (num_rings-1)*ring_gap, modules_per_ring * module_bin_x), dtype=np.int32)

    with open(lm_path, "rb") as f:
        raw_header = f.read(header_size)
        header = lm_header(raw_header)
        isv2 = header.is_v2()
        if force_v2: isv2 = True
        if force_v1: isv2 = False
        evt_size = size_v2 if isv2 else size_v1
        f.seek(0, os.SEEK_END)
        n_events = max(0, (f.tell() - header_size) // evt_size)
        f.seek(header_size)
        for _ in tqdm(range(n_events), desc="Parsing events"):
            raw = f.read(evt_size)
            if len(raw) < evt_size:
                break
            try:
                ev = parse_event_v2(raw) if isv2 else parse_event_v1(raw)
            except Exception:
                continue
            pair = int(ev["pair"]) 
            amount = ev["amount"]
            if (pair & singles_flag) != 0: continue
            if pair not in pair_map: continue
            mA, mB = pair_map[pair]
            x1_cr, y1_cr = int(ev["x1"]), int(ev["y1"])
            x2_cr, y2_cr = int(ev["x2"]), int(ev["y2"])
            local_y1_bin = y1_cr // rebin_y
            local_y2_bin = y2_cr // rebin_y
            local_x1_bin = x1_cr // rebin_x
            local_x2_bin = x2_cr // rebin_x
            X1, Y1, ring1 = detector_xy_mm(mA, local_x1_bin) 
            X2, Y2, ring2 = detector_xy_mm(mB, local_x2_bin)
            vt = lor_view(X1, Y1, X2, Y2)
            if vt is None: continue
            view_idx_inf, view_idx_sup, w_view_inf, w_view_sup, tang_inf,tang_sup,w_tang_inf,w_tang_sup, swap = vt
            
            if swap == 0:
                global_y1 = (ring1 * module_size + (ring1*ring_gap) + module_size - module_size*local_y1_bin/module_bin_y)
                global_y2 = (ring2 * module_size + (ring2*ring_gap) + module_size - module_size*local_y2_bin/module_bin_y)
            else:
                global_y2 = (ring1 * module_size + (ring1*ring_gap) + module_size - module_size*local_y1_bin/module_bin_y)
                global_y1 = (ring2 * module_size + (ring2*ring_gap) + module_size - module_size*local_y2_bin/module_bin_y)
             
            ring_dif = global_y2 - global_y1
            
            if ring_dif <= 0:
                bin_axial = int(sum(range(int(num_rings*module_size+(num_rings-1)*ring_gap+ring_dif))) + global_y2)
            else:
                bin_axial = int(bins_axial - (sum(range(int(num_rings*module_size+(num_rings-1)*ring_gap-ring_dif)))) - (int(num_rings*module_size+(num_rings-1)*ring_gap-ring_dif-global_y1-1))) -1
            
             
            sino[bin_axial, view_idx_inf, tang_inf] += w_view_inf*w_tang_inf*amount
            sino[bin_axial, view_idx_sup, tang_inf] += w_view_sup*w_tang_inf*amount
            sino[bin_axial, view_idx_inf, tang_sup] += w_view_inf*w_tang_sup*amount
            sino[bin_axial, view_idx_sup, tang_sup] += w_view_sup*w_tang_sup*amount
                    
            # hitmap
            ringA, modA = module_ring(mA)
            ringB, modB = module_ring(mB)
            reb_x1 = int(modA * module_bin_x + x1_cr // rebin_x)
            reb_y1 = int(ringA * module_bin_y + y1_cr // rebin_y)
            reb_x2 = int(modB * module_bin_x + x2_cr // rebin_x)
            reb_y2 = int(ringB * module_bin_y + y2_cr // rebin_y)
            if 0 <= reb_x1 < modules_per_ring * module_bin_x and 0 <= reb_y1 < num_rings * module_bin_y:
                hitmap[reb_y1, reb_x1] += 1
            if 0 <= reb_x2 < modules_per_ring * module_bin_x and 0 <= reb_y2 < num_rings * module_bin_y:
                hitmap[reb_y2, reb_x2] += 1
    return sino, hitmap, header            
                

import os
import struct
import numpy as np

# ---------------------------
# STIR .s/.hs
# ---------------------------
def write_stir_files(outbase, sino):
    """
    Write STIR-compatible sinogram and header files (.s / .hs)
    sino shape expected as [segments, axial, bins_views, tangential]
    """

    sfile = outbase + ".s"
    hsfile = outbase + ".hs"
    s_direct = outbase + "_directos.s"
    # --- Write .s binary ---
    with open(sfile, "wb") as f:
        sino.astype(np.float32).tofile(f)
    with open(s_direct, "wb") as f:
        sino[bins_axial//2-75:bins_axial//2+75,:,:].astype(np.float32).tofile(f)    
    # --- Write .hs header ---
    seg_min = [i - (segments // 2) for i in range(segments)]
    seg_max = seg_min
    ax_coord =[i+1 if i < (num_rings*module_bin_y+ (num_rings-1)*3.0) else int(2*(num_rings*module_bin_y+ (num_rings-1)*3.0)-i-1) for i in range(segments)]

    with open(hsfile, "w") as f:
        f.write("!INTERFILE :=\n")
        f.write("!imaging modality := PT\n")
        f.write(f"name of data file := {os.path.basename(sfile)}\n")
        f.write("originating system := Albira\n")
        f.write("!version of keys := STIR3.0\n")
        f.write("!GENERAL DATA :=\n")
        f.write("!GENERAL IMAGE DATA :=\n")
        f.write("!type of data := PET\n")
        f.write("imagedata byte order := LITTLEENDIAN\n")
        f.write("!PET STUDY (General) :=\n")
        f.write("!PET data type := Emission \n")
        f.write("applied corrections := {arc correction} \n")
        f.write("!number format := float\n")
        f.write("!number of bytes per pixel := 4\n")
        f.write("number of dimensions := 4\n")
        f.write("matrix axis label [4] := segment\n")
        f.write(f"!matrix size [4] := {segments}\n")
        f.write("matrix axis label [3] := axial coordinate\n")
        f.write("!matrix size [3] := { " + ",".join(map(str, ax_coord)) + " }\n")
        f.write("matrix axis label [2] := view\n")
        f.write(f"!matrix size [2] := {sino.shape[1]}\n")
        f.write("matrix axis label [1] := tangential coordinate\n")
        f.write(f"!matrix size [1] := {sino.shape[2]}\n")

        f.write("minimum ring difference per segment := { " + ",".join(map(str, seg_min)) + " }\n")
        f.write("maximum ring difference per segment := { " + ",".join(map(str, seg_max)) + " }\n")

        f.write("Scanner parameters :=\n")
        f.write("Scanner type := Albira\n")
        f.write(f"Number of rings := {int(num_rings*module_bin_y+ (num_rings-1)*3.0)}\n")
        f.write(f"Number of detectors per ring := {module_bin_x * modules_per_ring}\n")
        f.write(f"Inner ring diameter (cm) := {ring_radius*2/10.0:.4f}\n")
        f.write("Average depth of interaction (cm) := 0.84 \n")
        f.write(f"Distance between rings (cm) := {axial_fov/10/(module_size*num_rings + (num_rings-1)*3.0):.4f}\n")
        f.write(f"Default bin size (cm) := {transaxial_fov / bins_tang/10.0:.4f}\n")
        f.write("View offset (degrees) := 0\n")
        f.write("Maximum number of non-arc-corrected bins := 0\n")
        f.write(f"Default number of arc-corrected bins := {sino.shape[2]}\n")
        f.write(f"Energy_resolution := {energy_resol}\n")
        f.write("Reference energy (in keV) := 511\n")
        f.write("Number of blocks per bucket in transaxial direction := 1\n")
        f.write("Number of blocks per bucket in axial direction := 1\n")
        f.write("Number of crystals per block in axial direction := 1\n")
        f.write("Number of crystals per block in transaxial direction := 1\n")
        f.write("Number of crystals per singles unit in axial direction := 1\n")
        f.write("Number of crystals per singles unit in transaxial direction := 1\n")
        f.write("Scanner geometry := Cylindrical\n")
        f.write(f"Distance between crystals in axial direction (cm) := {axial_fov/10/(module_size*num_rings + (num_rings-1)*3.0):.4f}\n")
        f.write(f"Distance between crystals in transaxial direction (cm) := {transaxial_fov /bins_tang/10.0:.4f}\n")
        f.write(f"Distance between blocks in axial direction (cm) := {axial_fov/10/(module_size*num_rings + (num_rings-1)*3.0):.4f}\n")
        f.write(f"Distance between blocks in transaxial direction (cm) := {transaxial_fov / bins_tang/10.0:.4f}\n")

        f.write("end scanner parameters :=\n")
        f.write(f"effective central bin size (cm) := {transaxial_fov /bins_tang/10.0:.4f}\n")
        f.write("number of time frames := 1\n")
        f.write("start vertical bed position (mm) := 0\n")
        f.write("start horizontal bed position (mm) := 0\n")
        f.write("!END OF INTERFILE :=\n")

    print(f"✅ STIR sinogram files created: {sfile} and {hsfile}")


# ---------------------------
# Slicer
# ---------------------------
def inspect_sinogram_slices(data):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    slice_idx = 0
    img_disp = ax.imshow(np.log1p(data[slice_idx]), cmap='hot', origin='lower')
    ax.set_title(f"Segment {slice_idx}")
    plt.colorbar(img_disp, ax=ax, label='log(counts)')
    axslice = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(axslice, 'Segment', 0, data.shape[0]-1, valinit=slice_idx, valstep=1)
    def update(val):
        idx = int(slider.val)
        img_disp.set_data(np.log1p(data[idx]))
        ax.set_title(f"Segment {idx}")
        fig.canvas.draw_idle()
    slider.on_changed(update)
    plt.show()

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lmfile")
    parser.add_argument("outbase")
    args = parser.parse_args()
    sino, hitmap, header = build_sinogram_and_hitmap(args.lmfile)
    
    #data = save_sinogram_analyze(args.outbase, sino)
    print(sino.shape)
    write_stir_files(outbase=args.outbase,sino=sino)

    counts_axial = np.sum(sino, axis=1).sum(axis=1)
    plt.plot(counts_axial)
    plt.title("AXIAL")
    plt.show()
    
    counts_tangencial = np.sum(sino, axis=0).sum(axis=0)
    plt.plot(counts_tangencial)
    plt.title("TANGENCIAL")
    plt.show()
    
    counts_views = np.sum(sino, axis=0).sum(axis=1)
    plt.plot(counts_views)
    plt.title("VIEWS")
    plt.show()
    
    plt.imshow(hitmap, origin="lower", cmap="hot")
    plt.title("Modulos rebineados (hitmap)")
    plt.colorbar(label="counts")
    plt.show()
    
    inspect_sinogram_slices(sino)
   
if __name__ == "__main__":
    main()
