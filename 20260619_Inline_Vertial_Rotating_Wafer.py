"""
AZ Thin Film Research - Multi-Pass Production Process Simulator
-------------------------------------------------------------------------------
Features an advanced 3D Analytical integration of a Cylindrical Target.
Models the dual-leg magnetic racetrack, sputter angles, plasma width, and target OD.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Force headless web rendering

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle

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

if os.path.exists("logo.png"): st.logo("logo.png", link="https://www.azthinfilm.com")
elif os.path.exists("logo.jpg"): st.logo("logo.jpg", link="https://www.azthinfilm.com")

# --- 2. Advanced 3D Cylindrical Mathematics Engine ---
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

def deposition_rate(x, y, L, amp, skew, theta_s_deg, rt_width_deg, t_mat, d_surf):
    """
    Exact analytic integration of Lambertian emission from a cylindrical racetrack.
    Accounts for target OD, sputter angle, and racetrack plasma width.
    """
    R_tube = 132.5 / 2.0
    R_t = R_tube + t_mat
    D_axis = d_surf + R_t  
    
    alpha = np.radians(theta_s_deg)
    w = np.radians(rt_width_deg)
    
    # Model Racetrack Width using 5 weighted integration points per magnetic leg
    if rt_width_deg > 0:
        offsets = np.linspace(-w/2, w/2, 5)
        weights = np.exp(-0.5 * (offsets / (w/4.0))**2)
        weights /= np.sum(weights)
    else:
        offsets, weights = np.array([0.0]), np.array([1.0])
        
    flux_total = np.zeros_like(x, dtype=float)
    
    # Non-uniformity polynomial coefficients (Lengthwise Bow/Skew)
    c0, c1, c2 = 1.0 + amp, 2.0 * skew / L, -4.0 * amp / (L**2)
    K0, K1, K2 = c0 + c1 * y + c2 * (y**2), c1 + 2.0 * c2 * y, c2
    v2, v1 = (L / 2.0) - y, (-L / 2.0) - y
    
    for offset, weight in zip(offsets, weights):
        for base_angle in [alpha, -alpha]:
            theta = base_angle + offset
            
            # 3D Origin Point of the emission strip on the cylinder
            P_x = R_t * np.sin(theta)
            P_z = D_axis - R_t * np.cos(theta)
            
            # Projection of substrate vector against the cylindrical surface normal
            C_proj = (x - P_x) * np.sin(theta) + P_z * np.cos(theta)
            M = np.maximum(C_proj, 0.0) * P_z  # Prevents backward emission through target body
            
            B2 = (x - P_x)**2 + P_z**2
            B = np.sqrt(B2)
            
            # Exact Analytic Integral of Lambertian line emission (1/r^4 formulation)
            sq_v2, sq_v1 = B2 + v2**2, B2 + v1**2
            arc_v2, arc_v1 = np.arctan(v2 / B), np.arctan(v1 / B)
            
            I0_v2 = v2 / (2 * B2 * sq_v2) + arc_v2 / (2 * B2 * B)
            I0_v1 = v1 / (2 * B2 * sq_v1) + arc_v1 / (2 * B2 * B)
            I1_v2, I1_v1 = -1.0 / (2 * sq_v2), -1.0 / (2 * sq_v1)
            I2_v2 = arc_v2 / (2 * B) - v2 / (2 * sq_v2)
            I2_v1 = arc_v1 / (2 * B) - v1 / (2 * sq_v1)
            
            term2 = K0 * I0_v2 + K1 * I1_v2 + K2 * I2_v2
            term1 = K0 * I0_v1 + K1 * I1_v1 + K2 * I2_v1
            
            flux_total += weight * M * (term2 - term1)
            
    return np.maximum(flux_total, 0.0)

@st.cache_data
def compute_multipass_fast(L, v_x, rph, amp, skew, target_um, peak_rate_nm_s, wx, wy, grid_x, grid_y, theta_s_deg, rt_width_deg, t_mat, d_surf):
    """Vectorized Kinematic Phase Engine."""
    X_start, X_end = -1000.0, 1000.0
    t_pass = (X_end - X_start) / v_x
    omega = (rph / 3600.0) * 2 * np.pi 
    
    # Scale numerical integration to match the user's targeted Peak Rate (nm/s)
    x_test = np.linspace(-300, 300, 400)
    test_flux = deposition_rate(x_test, np.zeros_like(x_test), L, amp, skew, theta_s_deg, rt_width_deg, t_mat, d_surf)
    peak_idx = np.argmax(test_flux)
    base_peak_flux, x_peak = test_flux[peak_idx], x_test[peak_idx]
    if base_peak_flux == 0: base_peak_flux = 1.0
    scale = peak_rate_nm_s / base_peak_flux
    
    r_pts, theta_pts = np.sqrt(wx**2 + wy**2), np.arctan2(wy, wx)
    r_grid, theta_grid = np.sqrt(grid_x**2 + grid_y**2), np.arctan2(grid_y, grid_x)
    
    all_r = np.concatenate([r_pts.flatten(), r_grid.flatten()])
    unique_r, inv_idx = np.unique(np.round(all_r, 5), return_inverse=True)
    inv_idx_pts = inv_idx[:len(r_pts)]
    inv_idx_grid = inv_idx[len(r_pts):].reshape(grid_x.shape)
    
    num_phi = 360
    phi_arr = np.linspace(0, 2*np.pi, num_phi, endpoint=False)
    num_t = min(max(200, int(t_pass / 10.0)), 1000)
    t_fwd, dt = np.linspace(0, t_pass, num_t, retstep=True)
    
    cx_fwd, cx_bwd = X_start + v_x * t_fwd, X_end - v_x * t_fwd
    T_fwd, T_bwd = np.zeros((len(unique_r), num_phi)), np.zeros((len(unique_r), num_phi))
    
    for i, r in enumerate(unique_r):
        angles = phi_arr[:, None] + omega * t_fwd[None, :]
        cos_a, sin_a = np.cos(angles), np.sin(angles)
        T_fwd[i, :] = np.sum(deposition_rate(cx_fwd[None, :] + r * cos_a, r * sin_a, L, amp, skew, theta_s_deg, rt_width_deg, t_mat, d_surf), axis=1) * dt
        T_bwd[i, :] = np.sum(deposition_rate(cx_bwd[None, :] + r * cos_a, r * sin_a, L, amp, skew, theta_s_deg, rt_width_deg, t_mat, d_surf), axis=1) * dt

    T_fwd *= scale; T_bwd *= scale

    # Determine Required Passes dynamically
    dose_1pass_nm = np.mean(T_fwd[np.argmin(unique_r), :]) 
    target_nm = target_um * 1000.0
    N_passes = max(1, int(np.round(target_nm / dose_1pass_nm))) if dose_1pass_nm > 0 else 1
    
    phi_arr_padded = np.append(phi_arr, 2*np.pi)
    T_fwd_padded, T_bwd_padded = np.column_stack((T_fwd, T_fwd[:, 0])), np.column_stack((T_bwd, T_bwd[:, 0]))
    start_angles = (np.arange(N_passes) * omega * t_pass) % (2*np.pi)
    
    thick_pts = np.zeros(len(r_pts))
    for j in range(len(r_pts)):
        r_idx, theta = inv_idx_pts[j], theta_pts[j]
        dose_fwd = np.interp((theta + start_angles[0::2]) % (2*np.pi), phi_arr_padded, T_fwd_padded[r_idx, :])
        dose_bwd = np.interp((theta + start_angles[1::2]) % (2*np.pi), phi_arr_padded, T_bwd_padded[r_idx, :]) if len(start_angles) > 1 else [0]
        thick_pts[j] = np.sum(dose_fwd) + np.sum(dose_bwd)

    grid_flat_theta, inv_idx_grid_flat = theta_grid.flatten(), inv_idx_grid.flatten()
    thick_grid_flat = np.zeros(len(grid_flat_theta))
    
    for j in range(len(grid_flat_theta)):
        r_idx, theta = inv_idx_grid_flat[j], grid_flat_theta[j]
        dose_fwd = np.interp((theta + start_angles[0::2]) % (2*np.pi), phi_arr_padded, T_fwd_padded[r_idx, :])
        dose_bwd = np.interp((theta + start_angles[1::2]) % (2*np.pi), phi_arr_padded, T_bwd_padded[r_idx, :]) if len(start_angles) > 1 else [0]
        thick_grid_flat[j] = np.sum(dose_fwd) + np.sum(dose_bwd)
        
    thick_grid = thick_grid_flat.reshape(grid_x.shape)
    mean_th = np.mean(thick_pts)
    pts_norm = (thick_pts / mean_th) * 100.0 if mean_th > 0 else thick_pts * 0
    grid_norm = (thick_grid / mean_th) * 100.0 if mean_th > 0 else thick_grid * 0
    unif = (np.max(pts_norm) - np.min(pts_norm)) / 2.0 if mean_th > 0 else 0
    
    return pts_norm, grid_norm, unif, mean_th / 1000.0, N_passes, N_passes * t_pass, base_peak_flux, x_peak, thick_pts, thick_grid

def format_time(seconds):
    h, m = int(seconds // 3600), int((seconds % 3600) // 60)
    return f"{h}h {m}m" if h > 0 else f"{m}m"

def draw_rays(ax, alpha, width, R_t, d_surf, color, label):
    """Live CAD Rendering of the Magnetic Plasma Projection."""
    D_axis = R_t + d_surf
    a_rad, w_rad = np.radians(alpha), np.radians(width)
    for sign in [1, -1]:
        ang = sign * a_rad
        px, pz = R_t * np.sin(ang), D_axis - R_t * np.cos(ang)
        nx, nz = np.sin(ang), -np.cos(ang)
        ax.annotate('', xy=(px + 35*nx, pz + 35*nz), xytext=(px, pz),
                    arrowprops=dict(facecolor=color, edgecolor=color, width=2, headwidth=8))
        if width > 0:
            arcs = np.linspace(ang - w_rad/2, ang + w_rad/2, 20)
            xs, zs = R_t * np.sin(arcs), D_axis - R_t * np.cos(arcs)
            ax.plot(xs, zs, color=color, lw=5, label=label if sign == 1 else None)
        else:
            if sign == 1: ax.plot([], [], color=color, lw=5, label=label)

# --- 3. Streamlit User Interface ---
st.markdown('<h1 class="main-title">AZ Thin Film Research</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="sub-title">Production Solver: Cylindrical Target Kinematics</h3>', unsafe_allow_html=True)

wx, wy, grid_x, grid_y, R_wafer = setup_geometry()

with st.sidebar:
    if os.path.exists("logo.png"): st.image("logo.png", use_container_width=True)
    elif os.path.exists("logo.jpg"): st.image("logo.jpg", use_container_width=True)
    else: st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>AZ Thin Film</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'><a href='https://www.azthinfilm.com' target='_blank'>www.azthinfilm.com</a></p>", unsafe_allow_html=True)
    st.divider()

    st.header("1. Active Metrology View")
    view_mode = st.radio("Select process step:", ["Step 1: DC Base Layer", "Step 2: HiPIMS Top Layer", "Combined Final Stack"])
    
    st.header("2. Shared Hardware Geometry")
    t_mat = st.slider("Target Material Thickness (mm)", 2.0, 20.0, 10.0, 1.0)
    st.caption(f"Base Tube OD: 132.5 mm | **Total Target OD: {132.5 + 2*t_mat:.1f} mm**")
    d_surf = st.slider("Target-to-Substrate Dist. (mm)", 50, 300, 100, 5)
    L = st.slider("Cathode Length (mm)", 650, 900, 800, 10)
    R_t = (132.5 / 2.0) + t_mat
    
    st.header("3. Process Configurations")
    with st.expander("STEP 1: DC Base Layer (Bulk CrN)", expanded=True):
        t1 = st.number_input("Target Thickness (µm)", value=9.0, step=0.5, key="t1")
        p1 = st.number_input("Peak Target Rate (nm/s)", value=30.0, step=1.0, key="p1")
        v1 = st.slider("Translation Speed (mm/s)", 3.0, 100.0, 10.0, 1.0, key="v1")
        r1 = st.slider("Wafer Rotation (RPH)", 0.0, 10.0, 3.0, 0.1, key="r1")
        st.divider()
        st.caption("Magnetic Racetrack Config")
        alpha1 = st.slider("Sputter Angle (± degrees)", 10.0, 30.0, 12.0, 1.0, key="ang1")
        w1 = st.slider("Plasma Width (degrees)", 0.0, 45.0, 15.0, 1.0, key="w1")
        amp1 = st.slider("DC Profile Bow (%)", -50, 50, -5, 1, key="a1")
        skew1 = st.slider("DC Profile Skew (%)", -50, 50, 0, 1, key="s1")
        
    with st.expander("STEP 2: HiPIMS Top Layer (Cap CrN)", expanded=False):
        t2 = st.number_input("Target Thickness (µm)", value=1.0, step=0.1, key="t2")
        p2 = st.number_input("Peak Target Rate (nm/s)", value=10.0, step=1.0, key="p2")
        v2 = st.slider("Translation Speed (mm/s)", 3.0, 100.0, 10.0, 1.0, key="v2")
        r2 = st.slider("Wafer Rotation (RPH)", 0.0, 10.0, 3.0, 0.1, key="r2")
        st.divider()
        st.caption("Magnetic Racetrack Config")
        alpha2 = st.slider("Sputter Angle (± degrees)", 10.0, 30.0, 15.0, 1.0, key="ang2")
        w2 = st.slider("Plasma Width (degrees)", 0.0, 45.0, 25.0, 1.0, key="w2")
        amp2 = st.slider("HiPIMS Profile Bow (%)", -50, 50, -10, 1, key="a2")
        skew2 = st.slider("HiPIMS Profile Skew (%)", -50, 50, 0, 1, key="s2")

with st.spinner('Simulating 3D Cylindrical Phase-Shifting...'):
    if "Step 1" in view_mode:
        show_comb = False
        pts_norm, grid_norm, final_unif, final_mean, passes, total_time, base_c, x_peak, _, _ = compute_multipass_fast(
            L, v1, r1, amp1/100, skew1/100, t1, p1, wx, wy, grid_x, grid_y, alpha1, w1, t_mat, d_surf)
        disp_passes, act_amp, act_skew, act_rate, act_alpha, act_w = f"{passes:,}", amp1/100.0, skew1/100.0, p1, alpha1, w1
    elif "Step 2" in view_mode:
        show_comb = False
        pts_norm, grid_norm, final_unif, final_mean, passes, total_time, base_c, x_peak, _, _ = compute_multipass_fast(
            L, v2, r2, amp2/100, skew2/100, t2, p2, wx, wy, grid_x, grid_y, alpha2, w2, t_mat, d_surf)
        disp_passes, act_amp, act_skew, act_rate, act_alpha, act_w = f"{passes:,}", amp2/100.0, skew2/100.0, p2, alpha2, w2
    else:
        show_comb = True
        _, _, _, _, n1, time1, base_c1, _, pts1, grid1 = compute_multipass_fast(
            L, v1, r1, amp1/100, skew1/100, t1, p1, wx, wy, grid_x, grid_y, alpha1, w1, t_mat, d_surf)
        _, _, _, _, n2, time2, base_c2, _, pts2, grid2 = compute_multipass_fast(
            L, v2, r2, amp2/100, skew2/100, t2, p2, wx, wy, grid_x, grid_y, alpha2, w2, t_mat, d_surf)
        
        thick_pts, thick_grid = pts1 + pts2, grid1 + grid2
        final_mean = np.mean(thick_pts) / 1000.0
        pts_norm = (thick_pts / (final_mean * 1000.0)) * 100.0
        grid_norm = (thick_grid / (final_mean * 1000.0)) * 100.0
        final_unif = (np.max(pts_norm) - np.min(pts_norm)) / 2.0
        disp_passes, total_time = f"{n1:,} + {n2:,}", time1 + time2
        act_amp, act_skew, act_rate, act_alpha, act_w = amp1/100.0, skew1/100.0, p1, alpha1, w1 

# Dashboard Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Final Uniformity (+/-)", f"{final_unif:.2f} %")
col2.metric("Achieved Mean Thickness", f"{final_mean:.3f} µm")
col3.metric("Required Linear Passes", disp_passes)
col4.metric("Process Cycle Time", format_time(total_time))

if final_unif <= 1.0: st.success("✅ **TARGET ACHIEVED:** Rotary magnetics yield sub-1% uniformity.")
else: st.error("❌ **TARGET NOT MET:** Adjust Translation Speed, Target Dist, or Hardware Geometry.")
st.divider()

# --- 4. Render 6-Panel Matplotlib Charts ---
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.35)
dev = max(0.5, final_unif * 1.5) 

# [Chart 1: Target Cross-Section Physics Visualizer]
ax_geom = fig.add_subplot(gs[0, 0])
ax_geom.set_title("Target Cross-Section Physics", fontweight='bold')
ax_geom.set_aspect('equal')
D_axis = d_surf + R_t
ax_geom.set_xlim(-180, 180)
ax_geom.set_ylim(-30, D_axis + R_t + 30)

ax_geom.axhline(0, color='cyan', lw=3, label="Substrate Plane")
ax_geom.add_patch(Circle((0, D_axis), 132.5/2, fill=True, color='lightgray', label="132.5mm Base Tube"))
ax_geom.add_patch(Circle((0, D_axis), R_t, fill=False, color='#1E3A8A', lw=2, label="Target Material"))

if show_comb:
    draw_rays(ax_geom, alpha1, w1, R_t, d_surf, 'red', 'DC Racetrack')
    draw_rays(ax_geom, alpha2, w2, R_t, d_surf, 'dodgerblue', 'HiPIMS Racetrack')
else:
    draw_rays(ax_geom, act_alpha, act_w, R_t, d_surf, 'red', 'Active Racetrack')

ax_geom.axis('off')
ax_geom.legend(loc='lower center', fontsize=9)

# [Chart 2: Horizontal Profile - SHOWS RABBIT EARS]
ax_horiz = fig.add_subplot(gs[0, 1])
x_vals = np.linspace(-400, 400, 200)

if show_comb:
    rate_x1 = p1 * (deposition_rate(x_vals, np.zeros_like(x_vals), L, amp1/100, skew1/100, alpha1, w1, t_mat, d_surf) / base_c1)
    rate_x2 = p2 * (deposition_rate(x_vals, np.zeros_like(x_vals), L, amp2/100, skew2/100, alpha2, w2, t_mat, d_surf) / base_c2)
    rate_x = rate_x1 + rate_x2
    ax_horiz.plot(x_vals, rate_x1, color='red', ls='--', lw=1.5, label='DC Component')
    ax_horiz.plot(x_vals, rate_x2, color='dodgerblue', ls='--', lw=1.5, label='HiPIMS Component')
    ax_horiz.plot(x_vals, rate_x, color='#1E3A8A', lw=3, label='Combined Plume')
else:
    rate_x = act_rate * (deposition_rate(x_vals, np.zeros_like(x_vals), L, act_amp, act_skew, act_alpha, act_w, t_mat, d_surf) / base_c)
    ax_horiz.plot(x_vals, rate_x, color='#1E3A8A', lw=3, label='Horizontal Plume (Dual Lobe)')

ax_horiz.axvspan(-R_wafer, R_wafer, color='gray', alpha=0.15, label='Wafer Translation Path')
ax_horiz.set_xlim(-400, 400)
ax_horiz.set_ylim(0, max(0.001, np.max(rate_x)*1.15))
ax_horiz.set_title("Horizontal Plume Cross-Section", fontweight='bold')
ax_horiz.set_xlabel("Translation Axis X (mm)")
ax_horiz.set_ylabel("Deposition Rate (nm/s)")
ax_horiz.grid(True, linestyle='--', alpha=0.5)
ax_horiz.legend(loc='lower center', fontsize=9)

# [Chart 3: Wafer Map]
ax_map = fig.add_subplot(gs[0, 2])
ax_map.set_title("Accumulated Thickness Map", fontweight='bold')
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

# [Chart 4: Top-Down Deposition Plume]
ax_chamber = fig.add_subplot(gs[1, 0])
ax_chamber.set_title("Top-Down Deposition Plume", fontweight='bold')
ax_chamber.set_aspect('equal')
ax_chamber.set_xlim(-400, 400)
ax_chamber.set_ylim(-600, 600)
CX, CY = np.meshgrid(np.linspace(-400, 400, 100), np.linspace(-600, 600, 100))

if show_comb:
    rate_2d_1 = p1 * (deposition_rate(CX, CY, L, amp1/100, skew1/100, alpha1, w1, t_mat, d_surf) / base_c1)
    rate_2d_2 = p2 * (deposition_rate(CX, CY, L, amp2/100, skew2/100, alpha2, w2, t_mat, d_surf) / base_c2)
    rate_2d = rate_2d_1 + rate_2d_2
else:
    rate_2d = act_rate * (deposition_rate(CX, CY, L, act_amp, act_skew, act_alpha, act_w, t_mat, d_surf) / base_c)

chamber_contour = ax_chamber.contourf(CX, CY, rate_2d, levels=40, cmap='magma')
cax_chamber = ax_chamber.inset_axes([1.04, 0.0, 0.05, 1.0])
max_rate = np.max(rate_2d) if np.max(rate_2d) > 0 else 1.0
ticks_ch = np.linspace(0, max_rate, 5)
cbar_chamber = fig.colorbar(chamber_contour, cax=cax_chamber, ticks=ticks_ch)
cbar_chamber.set_label("Deposition Rate (nm/s)")
cbar_chamber.ax.set_yticklabels([f"{v:.1f}" for v in ticks_ch])

cylinder = Rectangle((-R_t, -L/2), 2*R_t, L, fill=False, edgecolor='white', linestyle=':', lw=1.5, alpha=0.6, label="Target Body")
ax_chamber.add_patch(cylinder)

if show_comb:
    x_rt1 = R_t * np.sin(np.radians(alpha1))
    ax_chamber.plot([-x_rt1, -x_rt1], [-L/2, L/2], 'r--', lw=2, alpha=0.8)
    ax_chamber.plot([x_rt1, x_rt1], [-L/2, L/2], 'r--', lw=2, alpha=0.8)
    x_rt2 = R_t * np.sin(np.radians(alpha2))
    ax_chamber.plot([-x_rt2, -x_rt2], [-L/2, L/2], color='dodgerblue', ls='--', lw=2, alpha=0.8)
    ax_chamber.plot([x_rt2, x_rt2], [-L/2, L/2], color='dodgerblue', ls='--', lw=2, alpha=0.8)
else:
    x_rt = R_t * np.sin(np.radians(act_alpha))
    ax_chamber.plot([-x_rt, -x_rt], [-L/2, L/2], 'w--', lw=2.0, alpha=0.9, label="Magnetic Racetrack")
    ax_chamber.plot([x_rt, x_rt], [-L/2, L/2], 'w--', lw=2.0, alpha=0.9)

ax_chamber.axhline(0, color='cyan', linestyle='--', alpha=0.8, lw=2, label="Wafer Path")
ax_chamber.legend(loc='upper right', fontsize=8)

# [Chart 5: Vertical Source Flux Profile]
ax_vert = fig.add_subplot(gs[1, 1])
y_vals = np.linspace(-600, 600, 200)

if show_comb:
    peak_x = x_vals[np.argmax(rate_x)] # Center evaluation at peak geometry
    rate_y1 = p1 * (deposition_rate(np.full_like(y_vals, peak_x), y_vals, L, amp1/100, skew1/100, alpha1, w1, t_mat, d_surf) / base_c1)
    rate_y2 = p2 * (deposition_rate(np.full_like(y_vals, peak_x), y_vals, L, amp2/100, skew2/100, alpha2, w2, t_mat, d_surf) / base_c2)
    rate_y = rate_y1 + rate_y2
    ax_vert.plot(y_vals, rate_y, color='#1E3A8A', lw=2.5, label='Combined Diffused Flux')
    ax_vert.legend(loc='lower center')
else:
    u_vals = 2 * y_vals / L
    mask = (y_vals >= -L/2) & (y_vals <= L/2)
    emission = np.zeros_like(y_vals)
    emission[mask] = 1.0 + act_amp * (1.0 - u_vals[mask]**2) + act_skew * u_vals[mask]
    ax_em = ax_vert.twinx()
    ax_em.plot(y_vals, emission, 'r--', lw=2, label='Target Erosion Shape')
    ax_em.set_ylabel("Source Emission Shape", color='r')
    ax_em.set_ylim(0, max(2.0, np.max(emission)*1.2))
    
    rate_y = act_rate * (deposition_rate(np.full_like(y_vals, x_peak), y_vals, L, act_amp, act_skew, act_alpha, act_w, t_mat, d_surf) / base_c)
    ax_vert.plot(y_vals, rate_y, color='#1E3A8A', lw=2.5, label='Diffused Flux (at peak)')
    lines, labels = ax_vert.get_legend_handles_labels()
    lines2, labels2 = ax_em.get_legend_handles_labels()
    ax_vert.legend(lines + lines2, labels + labels2, loc='lower center')

ax_vert.axvspan(-R_wafer, R_wafer, color='gray', alpha=0.15)
ax_vert.set_xlim(-600, 600)
ax_vert.set_ylim(0, max(0.001, np.max(rate_y)*1.1))
ax_vert.set_title("Vertical Source Flux Profile", fontweight='bold')
ax_vert.set_xlabel("Cathode Axis Y (mm)")
ax_vert.set_ylabel("Deposition Rate (nm/s)")
ax_vert.grid(True, linestyle='--', alpha=0.5)

# [Chart 6: 49-Point Scatter]
ax_stats = fig.add_subplot(gs[1, 2])
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
         ha='center', va='center', fontsize=12, color='#555555', style='italic', weight='bold')

st.pyplot(fig)
