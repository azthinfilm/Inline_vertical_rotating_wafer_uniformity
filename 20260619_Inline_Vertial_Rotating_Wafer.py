"""
AZ Thin Film Research - Multi-Pass Production Process Simulator
-------------------------------------------------------------------------------
Upgraded to simulate absolute multi-pass reciprocating kinematics.
Includes DC vs HiPIMS toggles, RPH rotation, and absolute thickness targeting.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Force headless web rendering

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

# --- 1. Page Configuration & Custom Branding ---
st.set_page_config(page_title="AZ Thin Film | Production Simulator", page_icon="⚙️", layout="wide")

st.markdown("""
    <style>
    .block-container { border-top: 5px solid #1E3A8A; padding-top: 2rem; }
    .main-title { color: #1E3A8A; font-weight: 800; margin-bottom: -10px; }
    .sub-title { color: #555555; font-weight: 400; margin-bottom: 20px; }
    [data-testid="stMetricValue"] { color: #1E3A8A; }
    </style>
""", unsafe_allow_html=True)

# Auto-detects your uploaded logo from GitHub
if os.path.exists("logo.png"):
    st.logo("logo.png", link="https://www.azthinfilm.com")
elif os.path.exists("logo.jpg"):
    st.logo("logo.jpg", link="https://www.azthinfilm.com")

# --- 2. Highly Optimized Multi-Pass Mathematics Engine ---
@st.cache_data
def setup_geometry():
    """Generates the arrays for the 49 points and the visual dense grid."""
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

@st.cache_data
def compute_multipass_fast(L, v_x, rph, amp, skew, target_um, peak_rate_nm_s, wx, wy, grid_x, grid_y):
    """Vectorized Kinematic Phase Engine (Capable of instantly solving thousands of passes)."""
    X_start, X_end = -1000.0, 1000.0
    t_pass = (X_end - X_start) / v_x
    omega = (rph / 3600.0) * 2 * np.pi  # Convert RPH to radians/second
    
    # Scale mathematical integration to match hardware Peak Rate (nm/s)
    base_center_flux = deposition_rate(np.array([0.0]), np.array([0.0]), L, 0.0, 0.0)[0]
    if base_center_flux == 0: base_center_flux = 1.0
    
    # Map polar coordinates to vastly speed up integration
    r_pts, theta_pts = np.sqrt(wx**2 + wy**2), np.arctan2(wy, wx)
    r_grid, theta_grid = np.sqrt(grid_x**2 + grid_y**2), np.arctan2(grid_y, grid_x)
    
    all_r = np.concatenate([r_pts.flatten(), r_grid.flatten()])
    unique_r, inv_idx = np.unique(np.round(all_r, 5), return_inverse=True)
    inv_idx_pts = inv_idx[:len(r_pts)]
    inv_idx_grid = inv_idx[len(r_pts):].reshape(grid_x.shape)
    
    # 1. Integrate one Forward and one Backward sweep covering all 360 degrees
    num_phi = 720
    phi_arr = np.linspace(0, 2*np.pi, num_phi, endpoint=False)
    num_t = max(200, int(t_pass / 10.0))  
    num_t = min(num_t, 1000)
    t_fwd, dt = np.linspace(0, t_pass, num_t, retstep=True)
    
    cx_fwd, cx_bwd = X_start + v_x * t_fwd, X_end - v_x * t_fwd
    T_fwd, T_bwd = np.zeros((len(unique_r), num_phi)), np.zeros((len(unique_r), num_phi))
    
    for i, r in enumerate(unique_r):
        angles = phi_arr[:, None] + omega * t_fwd[None, :]
        cos_a, sin_a = np.cos(angles), np.sin(angles)
        
        T_fwd[i, :] = np.sum(deposition_rate(cx_fwd[None, :] + r * cos_a, r * sin_a, L, amp, skew), axis=1) * dt
        T_bwd[i, :] = np.sum(deposition_rate(cx_bwd[None, :] + r * cos_a, r * sin_a, L, amp, skew), axis=1) * dt

    scale = peak_rate_nm_s / base_center_flux
    T_fwd *= scale
    T_bwd *= scale

    # 2. Determine Required Passes
    dose_1pass_nm = np.mean(T_fwd[np.argmin(unique_r), :]) 
    target_nm = target_um * 1000.0
    N_passes = max(1, int(np.round(target_nm / dose_1pass_nm))) if dose_1pass_nm > 0 else 1
    
    phi_arr_padded = np.append(phi_arr, 2*np.pi)
    T_fwd_padded = np.column_stack((T_fwd, T_fwd[:, 0]))
    T_bwd_padded = np.column_stack((T_bwd, T_bwd[:, 0]))
    
    # 3. Simulate Kinematic Phase-Shifts across all passes
    start_angles = (np.arange(N_passes) * omega * t_pass) % (2*np.pi)
    
    thick_pts = np.zeros(len(r_pts))
    for j in range(len(r_pts)):
        r_idx, theta = inv_idx_pts[j], theta_pts[j]
        dose_fwd = np.interp((theta + start_angles[0::2]) % (2*np.pi), phi_arr_padded, T_fwd_padded[r_idx, :])
        dose_bwd = np.interp((theta + start_angles[1::2]) % (2*np.pi), phi_arr_padded, T_bwd_padded[r_idx, :]) if len(start_angles) > 1 else [0]
        thick_pts[j] = np.sum(dose_fwd) + np.sum(dose_bwd)

    grid_flat_theta = theta_grid.flatten()
    inv_idx_grid_flat = inv_idx_grid.flatten()
    thick_grid_flat = np.zeros(len(grid_flat_theta))
    
    for j in range(len(grid_flat_theta)):
        r_idx, theta = inv_idx_grid_flat[j], grid_flat_theta[j]
        dose_fwd = np.interp((theta + start_angles[0::2]) % (2*np.pi), phi_arr_padded, T_fwd_padded[r_idx, :])
        dose_bwd = np.interp((theta + start_angles[1::2]) % (2*np.pi), phi_arr_padded, T_bwd_padded[r_idx, :]) if len(start_angles) > 1 else [0]
        thick_grid_flat[j] = np.sum(dose_fwd) + np.sum(dose_bwd)
        
    thick_grid = thick_grid_flat.reshape(grid_x.shape)
    
    mean_th = np.mean(thick_pts)
    thick_pts_norm = (thick_pts / mean_th) * 100.0 if mean_th > 0 else thick_pts * 0
    thick_grid_norm = (thick_grid / mean_th) * 100.0 if mean_th > 0 else thick_grid * 0
    unif = (np.max(thick_pts_norm) - np.min(thick_pts_norm)) / 2.0 if mean_th > 0 else 0
    time_sec = N_passes * t_pass
    
    return thick_pts_norm, thick_grid_norm, unif, mean_th / 1000.0, N_passes, time_sec, thick_pts, thick_grid

def format_time(seconds):
    h, m = int(seconds // 3600), int((seconds % 3600) // 60)
    return f"{h}h {m}m" if h > 0 else f"{m}m"

# --- 3. Streamlit User Interface ---
st.markdown('<h1 class="main-title">AZ Thin Film Research</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="sub-title">Production Solver: DC Base & HiPIMS Top Layer</h3>', unsafe_allow_html=True)

wx, wy, grid_x, grid_y, R_wafer = setup_geometry()

with st.sidebar:
    if os.path.exists("logo.png"): st.image("logo.png", use_container_width=True)
    elif os.path.exists("logo.jpg"): st.image("logo.jpg", use_container_width=True)
    else: st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>AZ Thin Film</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'><a href='https://www.azthinfilm.com' target='_blank'>www.azthinfilm.com</a></p>", unsafe_allow_html=True)
    
    st.header("1. Active Metrology View")
    view_mode = st.radio("Select which step to evaluate:", ["Step 1: DC Base Layer", "Step 2: HiPIMS Top Layer", "Combined Final Stack"])
    
    st.header("2. Process Configurations")
    with st.expander("STEP 1: DC Base Layer (Bulk CrN)", expanded=True):
        t1 = st.number_input("Target Thickness (µm)", value=9.0, step=0.5, key="t1")
        p1 = st.number_input("Peak Target Rate (nm/s)", value=30.0, step=1.0, key="p1")
        v1 = st.slider("Translation Speed (mm/s)", 3.0, 100.0, 10.0, 1.0, key="v1")
        r1 = st.slider("Wafer Rotation (RPH)", 0.0, 10.0, 3.0, 0.1, key="r1")
        amp1 = st.slider("DC Profile Bow (%)", -50, 50, -5, 1, key="a1")
        skew1 = st.slider("DC Profile Skew (%)", -50, 50, 0, 1, key="s1")
        
    with st.expander("STEP 2: HiPIMS Top Layer (Cap CrN)", expanded=False):
        t2 = st.number_input("Target Thickness (µm)", value=1.0, step=0.1, key="t2")
        p2 = st.number_input("Peak Target Rate (nm/s)", value=10.0, step=1.0, key="p2")
        v2 = st.slider("Translation Speed (mm/s)", 3.0, 100.0, 10.0, 1.0, key="v2")
        r2 = st.slider("Wafer Rotation (RPH)", 0.0, 10.0, 3.0, 0.1, key="r2")
        amp2 = st.slider("HiPIMS Profile Bow (%)", -50, 50, -10, 1, key="a2")
        skew2 = st.slider("HiPIMS Profile Skew (%)", -50, 50, 0, 1, key="s2")

    st.header("3. Shared Hardware Geometry")
    L = st.slider("Cathode Length (mm)", 650, 900, 800, 10)

with st.spinner('Simulating Kinematic Phase-Shifting & Multi-Pass Accumulation...'):
    if "Step 1" in view_mode:
        pts_norm, grid_norm, final_unif, final_mean, passes, total_time, _, _ = compute_multipass_fast(
            L, v1, r1, amp1/100, skew1/100, t1, p1, wx, wy, grid_x, grid_y)
        disp_passes, active_amp, active_skew, active_rate = f"{passes:,}", amp1/100.0, skew1/100.0, p1
    elif "Step 2" in view_mode:
        pts_norm, grid_norm, final_unif, final_mean, passes, total_time, _, _ = compute_multipass_fast(
            L, v2, r2, amp2/100, skew2/100, t2, p2, wx, wy, grid_x, grid_y)
        disp_passes, active_amp, active_skew, active_rate = f"{passes:,}", amp2/100.0, skew2/100.0, p2
    else:
        # Combined Stack Logic
        _, _, _, _, n1, time1, pts1, grid1 = compute_multipass_fast(L, v1, r1, amp1/100, skew1/100, t1, p1, wx, wy, grid_x, grid_y)
        _, _, _, _, n2, time2, pts2, grid2 = compute_multipass_fast(L, v2, r2, amp2/100, skew2/100, t2, p2, wx, wy, grid_x, grid_y)
        
        thick_pts = pts1 + pts2
        thick_grid = grid1 + grid2
        final_mean = np.mean(thick_pts) / 1000.0
        pts_norm = (thick_pts / (final_mean * 1000.0)) * 100.0
        grid_norm = (thick_grid / (final_mean * 1000.0)) * 100.0
        final_unif = (np.max(pts_norm) - np.min(pts_norm)) / 2.0
        
        disp_passes, total_time = f"{n1:,} + {n2:,}", time1 + time2
        active_amp, active_skew, active_rate = amp1/100.0, skew1/100.0, p1 # Show DC params for unified plot

# Dashboard Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Final Uniformity (+/-)", f"{final_unif:.2f} %")
col2.metric("Achieved Mean Thickness", f"{final_mean:.3f} µm")
col3.metric("Required Linear Passes", disp_passes)
col4.metric("Process Cycle Time", format_time(total_time))

if final_unif <= 1.0:
    st.success("✅ **TARGET ACHIEVED:** Process formulation yields sub-1% uniformity.")
else:
    st.error("❌ **TARGET NOT MET:** Uniformity > ±1.0%. Adjust Translation Speed or Rotation (RPH) to fix resonance.")
st.divider()

# --- 4. Render Matplotlib Charts ---
fig = plt.figure(figsize=(15, 9.5))
gs = GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.35)
dev = max(0.5, final_unif * 1.5) 

base_c = deposition_rate(np.array([0.0]), np.array([0.0]), L, 0, 0)[0]
if base_c == 0: base_c = 1.0

# [Chart 1] Flux Profile
ax_profile = fig.add_subplot(gs[0, 0])
y_vals = np.linspace(-600, 600, 200)
u_vals = 2 * y_vals / L
mask = (y_vals >= -L/2) & (y_vals <= L/2)
emission = np.zeros_like(y_vals)
emission[mask] = 1.0 + active_amp * (1.0 - u_vals[mask]**2) + active_skew * u_vals[mask]

ax_emission = ax_profile.twinx()
ax_emission.plot(y_vals, emission, 'r--', lw=2, label='Relative Target Erosion/Wear')
ax_emission.set_ylabel("Source Emission Shape", color='r')
ax_emission.set_ylim(0, max(2.0, np.max(emission)*1.2))

rate_y = active_rate * (deposition_rate(np.zeros_like(y_vals), y_vals, L, active_amp, active_skew) / base_c)
ax_profile.plot(y_vals, rate_y, color='#1E3A8A', lw=2.5, label='Diffused Flux at Wafer')
ax_profile.axvspan(-R_wafer, R_wafer, color='gray', alpha=0.2, label='Wafer Vertical Extent')
ax_profile.set_xlim(-600, 600)
ax_profile.set_ylim(0, max(0.001, np.max(rate_y)*1.1))
ax_profile.set_title("Vertical Source Flux Profile", fontweight='bold')
ax_profile.set_xlabel("Cathode Axis Y (mm)")
ax_profile.set_ylabel("Deposition Rate (nm/s)", color='#1E3A8A')
ax_profile.grid(True, linestyle='--', alpha=0.5)

lines, labels = ax_profile.get_legend_handles_labels()
lines2, labels2 = ax_emission.get_legend_handles_labels()
ax_profile.legend(lines + lines2, labels + labels2, loc='lower center')

# [Chart 2] Wafer Map
ax_map = fig.add_subplot(gs[0, 1])
ax_map.set_title(f"Accumulated Thickness Map", fontweight='bold')
ax_map.set_xlabel("Wafer X (mm)")
ax_map.set_ylabel("Wafer Y (mm)")

ax_map.set_aspect('equal')
ax_map.set_xlim(-160, 160)
ax_map.set_ylim(-160, 160)
ax_map.add_patch(Circle((0,0), R_wafer, fill=False, color='black', lw=2))

levels = np.linspace(100 - dev, 100 + dev, 40) 
contour = ax_map.contourf(grid_x, grid_y, grid_norm, levels=levels, cmap='viridis', extend='both')
ax_map.scatter(wx, wy, c='white', s=15, edgecolors='black', zorder=5)

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

ax_chamber.set_aspect('equal')
ax_chamber.set_xlim(-600, 600)
ax_chamber.set_ylim(-600, 600)

CX, CY = np.meshgrid(np.linspace(-600, 600, 80), np.linspace(-600, 600, 80))
rate_2d = active_rate * (deposition_rate(CX, CY, L, active_amp, active_skew) / base_c)
chamber_contour = ax_chamber.contourf(CX, CY, rate_2d, levels=40, cmap='magma')

cax_chamber = ax_chamber.inset_axes([1.04, 0.0, 0.05, 1.0])
max_rate = np.max(rate_2d) if np.max(rate_2d) > 0 else 1.0
ticks_ch = np.linspace(0, max_rate, 5)
cbar_chamber = fig.colorbar(chamber_contour, cax=cax_chamber, ticks=ticks_ch)
cbar_chamber.set_label("Deposition Rate (nm/s)")
cbar_chamber.ax.set_yticklabels([f"{v:.1f}" for v in ticks_ch])

ax_chamber.plot([0, 0], [-L/2, L/2], 'w-', lw=5, solid_capstyle='round', label="Target Width")
ax_chamber.axhline(0, color='cyan', linestyle='--', alpha=0.8, lw=2, label="Wafer Path")
ax_chamber.legend(loc='upper right')

# [Chart 4] 49-Point Scatter
ax_stats = fig.add_subplot(gs[1, 1])
pt_radii = np.sqrt(wx**2 + wy**2)
ax_stats.scatter(pt_radii, pts_norm, color='#1E3A8A', s=45, alpha=0.7, edgecolors='black')
ax_stats.axhline(100, color='gray', linestyle='--', lw=2)
ax_stats.set_title("49-Point Radial Analysis", fontweight='bold')
ax_stats.set_xlabel("Distance from Wafer Center (mm)")
ax_stats.set_ylabel("Normalized Thickness (%)")
ax_stats.set_xlim(-5, 155)
ax_stats.set_ylim(100 - dev, 100 + dev)
ax_stats.grid(True, linestyle='--', alpha=0.5)

# --- AZ THIN FILM BRANDING WATERMARK ---
fig.text(0.5, 0.02, 'Simulation provided by AZ Thin Film Research  |  www.azthinfilm.com', 
         ha='center', va='center', fontsize=11, color='#555555', style='italic', weight='bold')

st.pyplot(fig)
