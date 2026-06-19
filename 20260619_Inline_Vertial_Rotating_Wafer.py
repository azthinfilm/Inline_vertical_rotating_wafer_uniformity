"""
Web-Ready Deposition Uniformity Simulator
-------------------------------------------------------------------------------
Designed exclusively for Streamlit Web Hosting. 
Includes mathematical fixes for squished circular colorbars and garbled text.
"""

import matplotlib
# Explicitly force "headless" web rendering to prevent Qt5/Desktop crashes
matplotlib.use("Agg")  

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

# --- 1. Page Configuration ---
st.set_page_config(page_title="Deposition Uniformity Simulator", layout="wide")

# --- 2. Core Mathematical & Geometry Functions ---
@st.cache_data
def setup_geometry():
    """Generates the 300mm wafer grid and SEMI 49-point locations. Cached for web speed."""
    R_wafer, edge_exc = 150.0, 3.0
    R_max = R_wafer - edge_exc
    
    r1, r2, r3 = R_max * np.sqrt(5)/7, R_max * np.sqrt(17)/7, R_max * np.sqrt(37)/7
    radii_rings, points_per_ring = [0, r1, r2, r3], [1, 8, 16, 24]
    
    wx, wy = [], []
    for r, n in zip(radii_rings, points_per_ring):
        if n == 1:
            wx.append(0); wy.append(0)
        else:
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            wx.extend(r * np.cos(angles))
            wy.extend(r * np.sin(angles))
            
    grid_r, grid_theta = np.linspace(0, R_wafer, 40), np.linspace(0, 2*np.pi, 40)
    R_g, T_g = np.meshgrid(grid_r, grid_theta)
    
    return np.array(wx), np.array(wy), R_g * np.cos(T_g), R_g * np.sin(T_g), R_wafer

def deposition_rate(x, y, L, amp, skew, d=100.0):
    """Analytic spatial flux model for non-uniform linear cathode."""
    A2, A = x**2 + d**2, np.sqrt(x**2 + d**2)
    c0, c1, c2 = 1.0 + amp, 2.0 * skew / L, -4.0 * amp / (L**2)
    
    K0, K1, K2 = c0 + c1 * y + c2 * (y**2), c1 + 2.0 * c2 * y, c2
    v2, v1 = (L / 2.0) - y, (-L / 2.0) - y
    
    def evaluate_integral(v):
        sq = np.sqrt(A2 + v**2)
        return K0 * (v / (A2 * sq)) + K1 * (-1.0 / sq) + K2 * (np.arcsinh(v / A) - v / sq)

    flux = d * (evaluate_integral(v2) - evaluate_integral(v1))
    return np.maximum(flux, 0.0)

def compute_thickness(L, v_x, rpm, amp, skew, wx, wy, grid_x, grid_y):
    """Kinematic accumulation simulating the physical process over time."""
    X_start, X_end = -1000.0, 1000.0
    t_total = (X_end - X_start) / v_x
    omega = rpm * 2 * np.pi / 60.0
    revs = max((t_total * rpm / 60.0), 1)
    
    # 49 Points Integration Pass
    t_pts, dt_pts = np.linspace(0, t_total, min(int(revs * 72) + 500, 25000), retstep=True)
    cx_pts = X_start + v_x * t_pts
    cos_wt, sin_wt = np.cos(omega * t_pts)[:, None], np.sin(omega * t_pts)[:, None]
    
    px = cx_pts[:, None] + wx[None, :] * cos_wt - wy[None, :] * sin_wt
    py = wx[None, :] * sin_wt + wy[None, :] * cos_wt
    thick_pts = np.sum(deposition_rate(px, py, L, amp, skew), axis=0) * dt_pts
    
    # Visual Grid Integration Pass 
    t_g, dt_g = np.linspace(0, t_total, min(int(revs * 36) + 200, 4000), retstep=True)
    cx_g = X_start + v_x * t_g
    cos_wt_g, sin_wt_g = np.cos(omega * t_g)[:, None, None], np.sin(omega * t_g)[:, None, None]
    
    gx = cx_g[:, None, None] + grid_x[None, :, :] * cos_wt_g - grid_y[None, :, :] * sin_wt_g
    gy = grid_x[None, :, :] * sin_wt_g + grid_y[None, :, :] * cos_wt_g
    thick_grid = np.sum(deposition_rate(gx, gy, L, amp, skew), axis=0) * dt_g
    
    # Normalizations
    mean_th = np.mean(thick_pts)
    abs_dose = mean_th * 100.0
    thick_pts_norm = (thick_pts / mean_th) * 100.0
    thick_grid_norm = (thick_grid / mean_th) * 100.0
    unif = (np.max(thick_pts_norm) - np.min(thick_pts_norm)) / 2.0
    
    return thick_pts_norm, thick_grid_norm, unif, abs_dose

# --- 3. Streamlit User Interface ---
st.title("Rotary Cathode Deposition Uniformity Simulator")
st.markdown("Interactive quotation tool to evaluate translation and rotation kinematics for 300mm wafers.")

wx, wy, grid_x, grid_y, R_wafer = setup_geometry()

# HTML Native Sidebar Controls
with st.sidebar:
    st.header("System Kinematics")
    L = st.slider("Cathode Length (mm)", 650, 900, 800, 10)
    rpm = st.slider("Wafer Rotation (RPM)", 0, 120, 60, 5)
    v_x = st.slider("Translation Speed (mm/s)", 1, 30, 10, 1)

    st.header("Source Profile (Wear/Drift)")
    amp_pct = st.slider("Profile Bow/Amp (%) [- is Dogbone]", -50, 50, 0, 1)
    skew_pct = st.slider("Profile Skew (%) [+ is Top-Heavy]", -50, 50, 0, 1)

amp, skew = amp_pct / 100.0, skew_pct / 100.0

# Calculate Physics
with st.spinner('Simulating Kinematics...'):
    thick_pts_norm, thick_grid_norm, unif, abs_dose = compute_thickness(L, v_x, rpm, amp, skew, wx, wy, grid_x, grid_y)

# HTML Native Results Dashboard
col1, col2, col3, col4 = st.columns(4)
col1.metric("Uniformity (+/-)", f"{unif:.2f} %")
col2.metric("Absolute Dose", f"{abs_dose:.1f} a.u.")
col3.metric("Thickness Span", f"Min: {np.min(thick_pts_norm):.2f}% | Max: {np.max(thick_pts_norm):.2f}%")

if unif <= 1.0:
    col4.success("✅ TARGET ACHIEVED (≤ ±1.0%)")
else:
    col4.error("❌ TARGET NOT MET (> ±1.0%)")

st.divider()

# --- 4. Render Matplotlib Charts ---
fig = plt.figure(figsize=(15, 9.5))
gs = GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.35)
dev = max(0.5, unif * 1.5) 

# [Chart 1] Flux Profile
ax_profile = fig.add_subplot(gs[0, 0])
y_vals = np.linspace(-600, 600, 200)
u_vals = 2 * y_vals / L
mask = (y_vals >= -L/2) & (y_vals <= L/2)
emission = np.zeros_like(y_vals)
emission[mask] = 1.0 + amp * (1.0 - u_vals[mask]**2) + skew * u_vals[mask]

ax_emission = ax_profile.twinx()
ax_emission.plot(y_vals, emission, 'r--', lw=2, label='Source Emission I(y)')
ax_emission.set_ylabel("Source Emission", color='r')
ax_emission.set_ylim(0, max(2.0, np.max(emission)*1.2))

rate_y = deposition_rate(0, y_vals, L, amp, skew)
ax_profile.plot(y_vals, rate_y, 'b-', lw=2.5, label='Diffused Flux')
ax_profile.axvspan(-R_wafer, R_wafer, color='gray', alpha=0.2, label='Wafer Extent')
ax_profile.set_xlim(-600, 600)
ax_profile.set_ylim(0, max(0.001, np.max(rate_y)*1.1))
ax_profile.set_title("Vertical Source Profile", fontweight='bold')
ax_profile.set_xlabel("Cathode Axis Y (mm)")
ax_profile.set_ylabel("Flux Intensity", color='b')
ax_profile.grid(True, linestyle='--', alpha=0.5)

lines, labels = ax_profile.get_legend_handles_labels()
lines2, labels2 = ax_emission.get_legend_handles_labels()
ax_profile.legend(lines + lines2, labels + labels2, loc='lower center')

# [Chart 2] Wafer Map
ax_map = fig.add_subplot(gs[0, 1])
ax_map.set_title("Wafer Uniformity Map", fontweight='bold')
ax_map.set_xlabel("Wafer X (mm)")
ax_map.set_ylabel("Wafer Y (mm)")

# Lock plot shape to prevent layout squishing
ax_map.set_aspect('equal')
ax_map.set_xlim(-160, 160)
ax_map.set_ylim(-160, 160)
ax_map.add_patch(Circle((0,0), R_wafer, fill=False, color='black', lw=2))

levels = np.linspace(100 - dev, 100 + dev, 40) 
contour = ax_map.contourf(grid_x, grid_y, thick_grid_norm, levels=levels, cmap='viridis', extend='both')
ax_map.scatter(wx, wy, c='white', s=15, edgecolors='black', zorder=5)

# BULLETPROOF COLORBAR FIX
cax_map = ax_map.inset_axes([1.04, 0.0, 0.05, 1.0])
ticks_map = np.linspace(100 - dev, 100 + dev, 5)
cbar_map = fig.colorbar(contour, cax=cax_map, ticks=ticks_map)
cbar_map.set_label("Normalized Thickness (%)")
cbar_map.ax.set_yticklabels([f"{v:.2f}" for v in ticks_map])

# [Chart 3] Chamber Top-Down
ax_chamber = fig.add_subplot(gs[1, 0])
ax_chamber.set_title("Top-Down Deposition Plume", fontweight='bold')
ax_chamber.set_xlabel("Translation Axis X (mm)")
ax_chamber.set_ylabel("Cathode Axis Y (mm)")

# Lock limits
ax_chamber.set_aspect('equal')
ax_chamber.set_xlim(-600, 600)
ax_chamber.set_ylim(-600, 600)

CX, CY = np.meshgrid(np.linspace(-600, 600, 80), np.linspace(-600, 600, 80))
rate_2d = deposition_rate(CX, CY, L, amp, skew)
chamber_contour = ax_chamber.contourf(CX, CY, rate_2d, levels=40, cmap='magma')

# BULLETPROOF COLORBAR FIX
cax_chamber = ax_chamber.inset_axes([1.04, 0.0, 0.05, 1.0])
ticks_ch = np.linspace(0, np.max(rate_2d), 5)
cbar_chamber = fig.colorbar(chamber_contour, cax=cax_chamber, ticks=ticks_ch)
cbar_chamber.set_label("Relative Source Flux")
cbar_chamber.ax.set_yticklabels([f"{v:.3f}" for v in ticks_ch])

ax_chamber.plot([0, 0], [-L/2, L/2], 'w-', lw=5, solid_capstyle='round', label="Target Width")
ax_chamber.axhline(0, color='cyan', linestyle='--', alpha=0.8, lw=2, label="Wafer Path")
ax_chamber.legend(loc='upper right')

# [Chart 4] 49-Point Scatter
ax_stats = fig.add_subplot(gs[1, 1])
pt_radii = np.sqrt(wx**2 + wy**2)
ax_stats.scatter(pt_radii, thick_pts_norm, c='blue', s=45, alpha=0.7, edgecolors='black')
ax_stats.axhline(100, color='gray', linestyle='--', lw=2)
ax_stats.set_title("49-Point Radial Analysis", fontweight='bold')
ax_stats.set_xlabel("Distance from Wafer Center (mm)")
ax_stats.set_ylabel("Normalized Thickness (%)")
ax_stats.set_xlim(-5, 155)
ax_stats.set_ylim(100 - dev, 100 + dev)
ax_stats.grid(True, linestyle='--', alpha=0.5)

# Send Figure directly to Streamlit Web Page
st.pyplot(fig)
