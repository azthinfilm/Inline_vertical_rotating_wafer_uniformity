"""
AZ Thin Film Research - Multi-Pass Production Process Simulator
-------------------------------------------------------------------------------
Ultimate Multiphysics Edition: 
* Features exact Inter-pass Dwell Rotation indexing to break Kinematic Resonance.
* Fully continuous 3D unrolled racetrack integration.
* E x B Hall Current Drift & Asymmetric Anode Biasing.
"""

import os
import matplotlib
matplotlib.use("Agg")  

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle
from scipy.interpolate import RectBivariateSpline

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

try:
    if os.path.exists("logo.png"): st.logo("logo.png", link="https://www.azthinfilm.com")
    elif os.path.exists("logo.jpg"): st.logo("logo.jpg", link="https://www.azthinfilm.com")
except: pass

# --- 2. Advanced 3D Vectorized Racetrack Engine ---
@st.cache_data
def setup_geometry():
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

def generate_source_points(L, alpha_deg, w_deg, t_mat, amp, skew, lr_bias, hall_factor):
    """Generates the 3D differential point cloud for a completely continuous racetrack loop."""
    R_tube = 132.5 / 2.0
    R_t = R_tube + t_mat
    alpha_rad, w_rad = np.radians(alpha_deg), np.radians(w_deg)
    
    R_ta_base = min(R_t * alpha_rad, L/2.01)
    L_s = L - 2 * R_ta_base
    
    if w_deg > 0:
        offsets, weights = np.linspace(-R_t * w_rad / 2, R_t * w_rad / 2, 3), [0.25, 0.5, 0.25]
    else:
        offsets, weights = [0.0], [1.0]
        
    Px, Py, Pz, Nx, Ny, Nz, WI = [], [], [], [], [], [], []
    N_leg = max(20, int(150 * (L_s / L)))
    N_turn = max(15, int(150 * (np.pi * R_ta_base / L)))
    
    for off, wt in zip(offsets, weights):
        R_ta = max(R_ta_base + off, 1.0)
        
        # Straight Legs
        y_leg = np.linspace(-L_s/2, L_s/2, N_leg)
        dy = L_s / (N_leg - 1) if N_leg > 1 else 0
        for sign_x in [1, -1]:
            for y in y_leg:
                v = sign_x 
                u = 2 * y / L
                ta_boost = max(0.0, -amp * 2.0) * (abs(u)**6)
                I = max(0.0, 1.0 + amp*(1.0 - u**2) + skew*u + lr_bias*v + hall_factor*u*v + ta_boost)
                
                theta = (v * R_ta) / R_t
                Px.append(R_t * np.sin(theta)); Py.append(y); Pz.append(-R_t * np.cos(theta))
                Nx.append(np.sin(theta)); Ny.append(0.0); Nz.append(-np.cos(theta))
                WI.append(I * wt * dy)
                
        # Turnarounds
        d_gamma = np.pi / (N_turn - 1) if N_turn > 1 else 0
        dl = R_ta * d_gamma
        for sign_y, gamma_range in [(1, np.linspace(0, np.pi, N_turn)), (-1, np.linspace(np.pi, 2*np.pi, N_turn))]:
            for gam in gamma_range:
                v = np.cos(gam)
                y = sign_y * L_s/2 + R_ta * np.sin(gam)
                u = 2 * y / L
                ta_boost = max(0.0, -amp * 2.0) * (abs(u)**6)
                I = max(0.0, 1.0 + amp*(1.0 - u**2) + skew*u + lr_bias*v + hall_factor*u*v + ta_boost)
                
                theta = (R_ta * v) / R_t
                Px.append(R_t * np.sin(theta)); Py.append(y); Pz.append(-R_t * np.cos(theta))
                Nx.append(np.sin(theta)); Ny.append(0.0); Nz.append(-np.cos(theta))
                WI.append(I * wt * dl)
                
    return np.array(Px), np.array(Py), np.array(Pz), np.array(Nx), np.array(Ny), np.array(Nz), np.array(WI)

def calculate_flux(X, Y, Px, Py, Pz_chamber, Nx, Ny, Nz, WI, cos_n):
    X, Y = np.atleast_1d(X), np.atleast_1d(Y)
    flux = np.zeros(len(X), dtype=float)
    batch_size = 5000
    for i in range(0, len(X), batch_size):
        x_b, y_b = X[i:i+batch_size, None], Y[i:i+batch_size, None]
        dx, dy, dz = x_b - Px[None, :], y_b - Py[None, :], 0.0 - Pz_chamber[None, :]
        D2 = dx**2 + dy**2 + dz**2
        D = np.sqrt(D2)
        cos_emit = np.maximum((dx * Nx[None, :] + dy * Ny[None, :] + dz * Nz[None, :]) / D, 0.0)
        cos_sub = np.maximum(-dz / D, 0.0) 
        dF = WI[None, :] * (cos_emit ** cos_n) * cos_sub / D2
        flux[i:i+batch_size] = np.sum(dF, axis=1)
    return flux

def bake_flux_field(Px, Py, Pz_chamber, Nx, Ny, Nz, WI, cos_n):
    xg, yg = np.linspace(-1200, 1200, 400), np.linspace(-700, 700, 250)
    XG, YG = np.meshgrid(xg, yg)
    flux_flat = calculate_flux(XG.flatten(), YG.flatten(), Px, Py, Pz_chamber, Nx, Ny, Nz, WI, cos_n)
    return RectBivariateSpline(yg, xg, flux_flat.reshape(YG.shape))

@st.cache_data
def compute_multipass_fast(L, v_x, rph, dwell_deg, amp, skew, lr_bias, hall, target_um, peak_rate_nm_s, wx, wy, grid_x, grid_y, theta_s, w, t_mat, d_surf, cos_n):
    X_start, X_end = -1000.0, 1000.0
    t_pass = (X_end - X_start) / v_x
    omega = (rph / 3600.0) * 2 * np.pi 
    dwell_rad = np.radians(dwell_deg)
    
    R_t = (132.5 / 2.0) + t_mat
    D_axis = d_surf + R_t
    Px, Py, Pz, Nx, Ny, Nz, WI = generate_source_points(L, theta_s, w, t_mat, amp, skew, lr_bias, hall)
    Pz_chamber = D_axis + Pz
    
    interp_field = bake_flux_field(Px, Py, Pz_chamber, Nx, Ny, Nz, WI, cos_n)
    
    x_test = np.linspace(-300, 300, 400)
    test_flux = interp_field.ev(np.zeros_like(x_test), x_test)
    peak_idx = np.argmax(test_flux)
    base_peak_flux, x_peak = test_flux[peak_idx], x_test[peak_idx]
    if base_peak_flux <= 0: base_peak_flux = 1.0
    scale = peak_rate_nm_s / base_peak_flux
    
    r_pts, theta_pts = np.sqrt(wx**2 + wy**2), np.arctan2(wy, wx)
    r_grid, theta_grid = np.sqrt(grid_x**2 + grid_y**2), np.arctan2(grid_y, grid_x)
    all_r = np.concatenate([r_pts.flatten(), r_grid.flatten()])
    unique_r, inv_idx = np.unique(np.round(all_r, 5), return_inverse=True)
    inv_idx_pts, inv_idx_grid = inv_idx[:len(r_pts)], inv_idx[len(r_pts):].reshape(grid_x.shape)
    
    num_phi = 360
    phi_arr = np.linspace(0, 2*np.pi, num_phi, endpoint=False)
    num_t = min(max(200, int(t_pass / 10.0)), 600)
    t_fwd, dt = np.linspace(0, t_pass, num_t, retstep=True)
    
    cx_fwd, cx_bwd = X_start + v_x * t_fwd, X_end - v_x * t_fwd
    T_fwd, T_bwd = np.zeros((len(unique_r), num_phi)), np.zeros((len(unique_r), num_phi))
    
    for i, r in enumerate(unique_r):
        angles = phi_arr[:, None] + omega * t_fwd[None, :]
        cos_a, sin_a = np.cos(angles), np.sin(angles)
        
        X_fwd, Y_sub = cx_fwd[None, :] + r * cos_a, r * sin_a
        T_fwd[i, :] = np.sum(interp_field.ev(Y_sub.flatten(), X_fwd.flatten()).reshape(X_fwd.shape), axis=1) * dt
        X_bwd = cx_bwd[None, :] + r * cos_a
        T_bwd[i, :] = np.sum(interp_field.ev(Y_sub.flatten(), X_bwd.flatten()).reshape(X_bwd.shape), axis=1) * dt

    T_fwd *= scale; T_bwd *= scale

    dose_1pass_nm = np.mean(T_fwd[np.argmin(unique_r), :]) 
    N_passes = max(1, int(np.round((target_um * 1000.0) / dose_1pass_nm))) if dose_1pass_nm > 0 else 1
    
    # Phase Accumulation Logic (Applies the specific Dwell Rotation Offset between passes)
    phi_pd = np.append(phi_arr, 2*np.pi)
    start_angles = (np.arange(N_passes) * (omega * t_pass + dwell_rad)) % (2*np.pi)
    
    T_fwd_pd, T_bwd_pd = np.column_stack((T_fwd, T_fwd[:, 0])), np.column_stack((T_bwd, T_bwd[:, 0]))
    
    thick_pts = np.zeros(len(r_pts))
    for j in range(len(r_pts)):
        r_idx, theta = inv_idx_pts[j], theta_pts[j]
        d_fwd = np.interp((theta + start_angles[0::2]) % (2*np.pi), phi_pd, T_fwd_pd[r_idx, :])
        d_bwd = np.interp((theta + start_angles[1::2]) % (2*np.pi), phi_pd, T_bwd_pd[r_idx, :]) if len(start_angles) > 1 else [0]
        thick_pts[j] = np.sum(d_fwd) + np.sum(d_bwd)

    thick_grid_flat = np.zeros(len(theta_grid.flatten()))
    for j, (r_idx, theta) in enumerate(zip(inv_idx_grid.flatten(), theta_grid.flatten())):
        d_fwd = np.interp((theta + start_angles[0::2]) % (2*np.pi), phi_pd, T_fwd_pd[r_idx, :])
        d_bwd = np.interp((theta + start_angles[1::2]) % (2*np.pi), phi_pd, T_bwd_pd[r_idx, :]) if len(start_angles) > 1 else [0]
        thick_grid_flat[j] = np.sum(d_fwd) + np.sum(d_bwd)
        
    thick_grid = thick_grid_flat.reshape(grid_x.shape)
    mean_th = np.mean(thick_pts)
    pts_norm = (thick_pts / mean_th) * 100.0 if mean_th > 0 else thick_pts * 0
    grid_norm = (thick_grid / mean_th) * 100.0 if mean_th > 0 else thick_grid * 0
    unif = (np.max(pts_norm) - np.min(pts_norm)) / 2.0 if mean_th > 0 else 0
    
    # Accurate Cycle Time Calculation including Dwell waits
    if omega > 1e-9:
        t_dwell_sec = dwell_rad / omega
    else:
        # Assume a rapid 10 RPM mechanical indexer if the continuous RPH is zero
        t_dwell_sec = dwell_rad / (10.0 * 2 * np.pi / 60.0) 
        
    total_time = N_passes * t_pass + max(0, N_passes - 1) * t_dwell_sec
    
    return pts_norm, grid_norm, unif, mean_th / 1000.0, N_passes, total_time, base_peak_flux, x_peak, thick_pts, thick_grid, interp_field

def format_time(seconds):
    h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    if h > 0: return f"{h}h {m}m"
    elif m > 0: return f"{m}m {s}s"
    else: return f"{s}s"

def get_plot_flux(x_arr, y_arr, interp, scale, base_c):
    x_flat, y_flat = np.atleast_1d(x_arr).flatten(), np.atleast_1d(y_arr).flatten()
    return scale * (interp.ev(y_flat, x_flat).reshape(np.asarray(x_arr).shape) / base_c)

# --- 3. Streamlit User Interface ---
st.markdown('<h1 class="main-title">AZ Thin Film Research</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="sub-title">Production Solver: Multiphysics Target & Clocking Engine</h3>', unsafe_allow_html=True)

with st.expander("📚 **View Engineering Physics Documentation & Mathematical Assumptions**", expanded=False):
    st.markdown("""
    ### Thin Film Mathematical Architecture
    This simulator bypasses basic 1D line-source approximations and implements a **True 3D Vectorized Point-Cloud Engine** to map the exact geometry of a cylindrical rotary magnetron.

    #### 1. 3D Cylindrical Point-Cloud Integration
    Instead of assuming a 1D flat line, this engine mathematically unrolls the target cylinder into a 3D coordinate space. It projects the precise $(x, y, z)$ emission path of the dual-leg magnetic racetrack, including the geometric arc of both semi-circular turnarounds, taking into account the exact Outer Diameter ($R_{tube} + t_{mat}$).
    
    #### 2. Sputter Flux Collimation ($Cos^n \\theta$)
    Emission from any differential point on the racetrack is calculated using a collimation-adjusted Lambertian distribution:
    $$d\\Phi \\propto \\frac{\\cos^n(\\theta_{emit}) \\cos(\\theta_{sub})}{D^2} dA$$
    DC sputtering typically follows $n=1.0$ (diffuse), while highly-ionized HiPIMS plasmas can be sharply collimated ($n=2.0$ to $3.0$), physically narrowing the resulting plume.

    #### 3. Turnaround Phase Synchronization (Kinematic Clocking)
    The uniformity map explicitly tracks the absolute azimuthal phase ($\\theta$) of the wafer at every moment. During translation, the wafer rotates at the defined RPH. Once out of the plasma plume, the engine mathematically halts translation, applies the user-defined **End-of-Pass Dwell Index**, calculates the resulting time penalty, and sets the precise, shifted starting clock angle for the subsequent reverse pass. This completely eliminates phase-averaging errors associated with instantaneous turnaround assumptions and allows engineering control over destructive phase interference.
    
    #### 4. E x B Hall Current Drift Build-up
    Because magnetic field strength dips at the geometric turnarounds, secondary electron confinement is weakened. The engine models this using a continuous cross-term $+ H_{all} \\cdot (u \\cdot v)$. This accurately replicates how the Hall current ($E \\times B$) drift loses electrons at the ends and must "rebuild" the plasma density sequentially via the Townsend avalanche process as it travels down the straightaways, generating an **anti-parallel longitudinal emission gradient** between the left and right legs.
    
    #### 5. Asymmetric Anode Biasing (L/R Gradient)
    The localized electric field between the cathode and external anodes (or chamber walls) can warp plasma equipotential lines. The simulator introduces a transverse azimuthal gradient $+ B_{ias} \\cdot v$ to cleanly shift plasma density toward the left or right racetrack leg.
    
    #### 6. Continuous Dogbone Turnarounds
    Historical models simulate turnaround wear as sharp step-functions. This simulator employs a $C^0$ continuous geometric interpolation. When a negative **Profile Bow** is applied (representing "dog-bone" target wear), the engine scales a higher-order polynomial ($u^6$) to smoothly boost the apex of the turnarounds. This correctly eliminates visual discontinuities while simulating massive localized plasma trapping in the erosion trenches.
    """)

wx, wy, grid_x, grid_y, R_wafer = setup_geometry()

with st.sidebar:
    try:
        if os.path.exists("logo.png"): st.image("logo.png", use_container_width=True)
        elif os.path.exists("logo.jpg"): st.image("logo.jpg", use_container_width=True)
    except: pass
    
    st.markdown("<p style='text-align: center;'><a href='https://www.azthinfilm.com' target='_blank'>www.azthinfilm.com</a></p>", unsafe_allow_html=True)
    st.divider()

    st.header("1. Active Metrology View")
    view_mode = st.radio("Select process step:", ["Step 1: DC Base Layer", "Step 2: HiPIMS Top Layer", "Combined Final Stack"])
    if "Combined" in view_mode: st.info("Hardware physics graphs will display Step 1 & 2 merged overlays.")
    
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
        dwell1 = st.slider("Turnaround Dwell Rotation (deg)", 0, 355, 90, 5, key="dw1", help="Mechanical indexing applied at the end of the stroke before reversing pass direction.")
        st.divider()
        st.caption("Multiphysics Plasma Modifiers")
        alpha1 = st.slider("Sputter Angle (± degrees)", 10.0, 30.0, 12.0, 1.0, key="ang1")
        w1 = st.slider("Plasma Width (degrees)", 0.0, 45.0, 15.0, 1.0, key="w1")
        cos1 = st.slider("Plume Shape Factor (Cos^n)", 0.5, 5.0, 1.0, 0.1, key="c1")
        amp1 = st.slider("Profile Bow (%) [- is Dogbone]", -50, 50, -25, 1, key="a1")
        skew1 = st.slider("Profile Skew (%) [+ is Top-Heavy]", -50, 50, 0, 1, key="s1")
        lrbias1 = st.slider("L/R Anode Bias (%) [+ is Right Heavy]", -50, 50, 0, 1, key="lr1")
        hall1 = st.slider("Hall Drift Build-Up (%)", 0, 50, 20, 1, key="h1")
        
    with st.expander("STEP 2: HiPIMS Top Layer (Cap CrN)", expanded=False):
        t2 = st.number_input("Target Thickness (µm)", value=1.0, step=0.1, key="t2")
        p2 = st.number_input("Peak Target Rate (nm/s)", value=10.0, step=1.0, key="p2")
        v2 = st.slider("Translation Speed (mm/s)", 3.0, 100.0, 10.0, 1.0, key="v2")
        r2 = st.slider("Wafer Rotation (RPH)", 0.0, 10.0, 3.0, 0.1, key="r2")
        dwell2 = st.slider("Turnaround Dwell Rotation (deg)", 0, 355, 90, 5, key="dw2")
        st.divider()
        st.caption("Multiphysics Plasma Modifiers")
        alpha2 = st.slider("Sputter Angle (± degrees)", 10.0, 30.0, 15.0, 1.0, key="ang2")
        w2 = st.slider("Plasma Width (degrees)", 0.0, 45.0, 25.0, 1.0, key="w2")
        cos2 = st.slider("Plume Shape Factor (Cos^n)", 0.5, 5.0, 2.0, 0.1, key="c2")
        amp2 = st.slider("HiPIMS Profile Bow (%)", -50, 50, -10, 1, key="a2")
        skew2 = st.slider("HiPIMS Profile Skew (%)", -50, 50, 0, 1, key="s2")
        lrbias2 = st.slider("L/R Anode Bias (%)", -50, 50, 0, 1, key="lr2")
        hall2 = st.slider("Hall Drift Build-Up (%)", 0, 50, 10, 1, key="h2")

with st.spinner('Calculating Phase Offsets & 3D Multiphysics Fields...'):
    if "Step 1" in view_mode:
        show_comb = False
        pts_norm, grid_norm, final_unif, final_mean, passes, total_time, base_c, x_peak, _, _, act_interp = compute_multipass_fast(
            L, v1, r1, dwell1, amp1/100, skew1/100, lrbias1/100, hall1/100, t1, p1, wx, wy, grid_x, grid_y, alpha1, w1, t_mat, d_surf, cos1)
        disp_passes, act_amp, act_skew, act_rate, act_alpha, act_w, act_cos = f"{passes:,}", amp1/100.0, skew1/100.0, p1, alpha1, w1, cos1
        act_lr, act_hall = lrbias1/100.0, hall1/100.0
    elif "Step 2" in view_mode:
        show_comb = False
        pts_norm, grid_norm, final_unif, final_mean, passes, total_time, base_c, x_peak, _, _, act_interp = compute_multipass_fast(
            L, v2, r2, dwell2, amp2/100, skew2/100, lrbias2/100, hall2/100, t2, p2, wx, wy, grid_x, grid_y, alpha2, w2, t_mat, d_surf, cos2)
        disp_passes, act_amp, act_skew, act_rate, act_alpha, act_w, act_cos = f"{passes:,}", amp2/100.0, skew2/100.0, p2, alpha2, w2, cos2
        act_lr, act_hall = lrbias2/100.0, hall2/100.0
    else:
        show_comb = True
        _, _, _, _, n1, time1, bc1, _, pts1, grid1, interp1 = compute_multipass_fast(L, v1, r1, dwell1, amp1/100, skew1/100, lrbias1/100, hall1/100, t1, p1, wx, wy, grid_x, grid_y, alpha1, w1, t_mat, d_surf, cos1)
        _, _, _, _, n2, time2, bc2, _, pts2, grid2, interp2 = compute_multipass_fast(L, v2, r2, dwell2, amp2/100, skew2/100, lrbias2/100, hall2/100, t2, p2, wx, wy, grid_x, grid_y, alpha2, w2, t_mat, d_surf, cos2)
        
        thick_pts, thick_grid = pts1 + pts2, grid1 + grid2
        final_mean = np.mean(thick_pts) / 1000.0
        pts_norm = (thick_pts / (final_mean * 1000.0)) * 100.0
        grid_norm = (thick_grid / (final_mean * 1000.0)) * 100.0
        final_unif = (np.max(pts_norm) - np.min(pts_norm)) / 2.0
        disp_passes, total_time = f"{n1:,} + {n2:,}", time1 + time2
        act_amp, act_skew, act_rate, act_alpha, act_w, act_cos = amp1/100.0, skew1/100.0, p1, alpha1, w1, cos1 
        act_lr, act_hall = lrbias1/100.0, hall1/100.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Final Uniformity (+/-)", f"{final_unif:.2f} %")
col2.metric("Achieved Mean Thickness", f"{final_mean:.3f} µm")
col3.metric("Required Linear Passes", disp_passes)
col4.metric("Process Cycle Time", format_time(total_time))

if final_unif <= 1.0: st.success("✅ **TARGET ACHIEVED:** Multiphysics rotary simulation yields sub-1% uniformity.")
else: st.error("❌ **TARGET NOT MET:** Adjust Rotation RPH or Dwell Angle to break Kinematic Resonance.")
st.divider()

# --- 4. Render 6-Panel Matplotlib Charts ---
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.35)
dev = max(0.5, final_unif * 1.5) 

# [Chart 1: Target Cross-Section]
ax_geom = fig.add_subplot(gs[0, 0])
ax_geom.set_title("Target Cross-Section Physics", fontweight='bold')
ax_geom.set_aspect('equal')
D_axis = d_surf + R_t
ax_geom.set_xlim(-180, 180)
ax_geom.set_ylim(-30, D_axis + R_t + 30)

ax_geom.axhline(0, color='cyan', lw=3, label="Substrate Plane")
ax_geom.add_patch(Circle((0, D_axis), 132.5/2, fill=True, color='lightgray', label="132.5mm Base Tube"))
ax_geom.add_patch(Circle((0, D_axis), R_t, fill=False, color='#1E3A8A', lw=2, label="Target Material"))

def draw_rays(ax, alpha, width, R_t, D_axis, color, label):
    a_rad, w_rad = np.radians(alpha), np.radians(width)
    for sign in [1, -1]:
        ang = sign * a_rad
        px, pz = R_t * np.sin(ang), D_axis - R_t * np.cos(ang)
        nx, nz = np.sin(ang), -np.cos(ang)
        ax.annotate('', xy=(px + 35*nx, pz + 35*nz), xytext=(px, pz), arrowprops=dict(facecolor=color, edgecolor=color, width=2, headwidth=8))
        if width > 0:
            arcs = np.linspace(ang - w_rad/2, ang + w_rad/2, 20)
            ax.plot(R_t * np.sin(arcs), D_axis - R_t * np.cos(arcs), color=color, lw=5, label=label if sign == 1 else None)
        elif sign == 1: ax.plot([], [], color=color, lw=5, label=label)

if show_comb:
    draw_rays(ax_geom, alpha1, w1, R_t, D_axis, 'red', 'DC Racetrack')
    draw_rays(ax_geom, alpha2, w2, R_t, D_axis, 'dodgerblue', 'HiPIMS Racetrack')
else:
    draw_rays(ax_geom, act_alpha, act_w, R_t, D_axis, 'red', 'Active Racetrack')

ax_geom.axis('off')
ax_geom.legend(loc='lower center', fontsize=9)

# [Chart 2: Horizontal Plume (Batman Curve)]
ax_horiz = fig.add_subplot(gs[0, 1])
x_vals = np.linspace(-400, 400, 200)

if show_comb:
    rate_x1 = get_plot_flux(x_vals, np.zeros_like(x_vals), interp1, p1, bc1)
    rate_x2 = get_plot_flux(x_vals, np.zeros_like(x_vals), interp2, p2, bc2)
    ax_horiz.plot(x_vals, rate_x1, color='red', ls='--', lw=1.5, label='DC Component')
    ax_horiz.plot(x_vals, rate_x2, color='dodgerblue', ls='--', lw=1.5, label='HiPIMS Component')
    ax_horiz.plot(x_vals, rate_x1 + rate_x2, color='#1E3A8A', lw=3, label='Combined Plume')
else:
    rate_x = get_plot_flux(x_vals, np.zeros_like(x_vals), act_interp, act_rate, base_c)
    ax_horiz.plot(x_vals, rate_x, color='#1E3A8A', lw=3, label=f'Plume Lobe (Cos n={act_cos:.1f})')

ax_horiz.axvspan(-R_wafer, R_wafer, color='gray', alpha=0.15, label='Wafer Translation Path')
ax_horiz.set_xlim(-400, 400)
ax_horiz.set_ylim(0, max(0.001, np.max(rate_x1 + rate_x2 if show_comb else rate_x)*1.15))
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

# [Chart 4: Top-Down Deposition Plume & Turnaround Visuals]
ax_chamber = fig.add_subplot(gs[1, 0])
ax_chamber.set_title("Top-Down Deposition Plume", fontweight='bold')
ax_chamber.set_aspect('equal')
ax_chamber.set_xlim(-400, 400)
ax_chamber.set_ylim(-600, 600)
CX, CY = np.meshgrid(np.linspace(-400, 400, 80), np.linspace(-600, 600, 80))

if show_comb:
    rate_2d = get_plot_flux(CX, CY, interp1, p1, bc1) + get_plot_flux(CX, CY, interp2, p2, bc2)
else:
    rate_2d = get_plot_flux(CX, CY, act_interp, act_rate, base_c)

chamber_contour = ax_chamber.contourf(CX, CY, rate_2d, levels=40, cmap='magma')
cax_chamber = ax_chamber.inset_axes([1.04, 0.0, 0.05, 1.0])
max_rate = np.max(rate_2d) if np.max(rate_2d) > 0 else 1.0
ticks_ch = np.linspace(0, max_rate, 5)
cbar_chamber = fig.colorbar(chamber_contour, cax=cax_chamber, ticks=ticks_ch)
cbar_chamber.set_label("Deposition Rate (nm/s)")
cbar_chamber.ax.set_yticklabels([f"{v:.1f}" for v in ticks_ch])

cylinder = Rectangle((-R_t, -L/2), 2*R_t, L, fill=False, edgecolor='white', linestyle=':', lw=1.5, alpha=0.6, label="Target Body")
ax_chamber.add_patch(cylinder)

def draw_racetrack_topdown(ax, L, alpha_deg, R_t, color, ls):
    alpha_rad = np.radians(alpha_deg)
    R_ta = min(R_t * alpha_rad, L/2.01)
    L_s = L - 2 * R_ta
    x_rt = R_t * np.sin(alpha_rad) 
    
    ax.plot([x_rt, x_rt], [-L_s/2, L_s/2], color=color, ls=ls, lw=2, alpha=0.9, label="Continuous Racetrack")
    ax.plot([-x_rt, -x_rt], [-L_s/2, L_s/2], color=color, ls=ls, lw=2, alpha=0.9)
    gamma = np.linspace(0, np.pi, 30)
    ax.plot(R_t * np.sin((R_ta * np.cos(gamma)) / R_t), L_s/2 + R_ta * np.sin(gamma), color=color, ls=ls, lw=2, alpha=0.9)
    gamma = np.linspace(np.pi, 2*np.pi, 30)
    ax.plot(R_t * np.sin((R_ta * np.cos(gamma)) / R_t), -L_s/2 + R_ta * np.sin(gamma), color=color, ls=ls, lw=2, alpha=0.9)

if show_comb:
    draw_racetrack_topdown(ax_chamber, L, alpha1, R_t, 'red', '--')
    draw_racetrack_topdown(ax_chamber, L, alpha2, R_t, 'dodgerblue', '--')
else:
    draw_racetrack_topdown(ax_chamber, L, act_alpha, R_t, 'white', '--')

ax_chamber.axhline(0, color='cyan', linestyle='--', alpha=0.8, lw=2, label="Wafer Path")
ax_chamber.legend(loc='upper right', fontsize=8)

# [Chart 5: Vertical Source Flux Profile (SMOOTH TURNAROUNDS)]
ax_vert = fig.add_subplot(gs[1, 1])
y_vals = np.linspace(-600, 600, 200)

if show_comb:
    peak_x = x_vals[np.argmax(rate_x1 + rate_x2)] 
    rate_y1 = get_plot_flux(np.full_like(y_vals, peak_x), y_vals, interp1, p1, bc1)
    rate_y2 = get_plot_flux(np.full_like(y_vals, peak_x), y_vals, interp2, p2, bc2)
    rate_y = rate_y1 + rate_y2
    ax_vert.plot(y_vals, rate_y, color='#1E3A8A', lw=2.5, label='Combined Diffused Flux')
    ax_vert.legend(loc='lower center')
else:
    u_vals = np.clip(2 * y_vals / L, -1.0, 1.0)
    
    ta_boost = np.maximum(0.0, -act_amp * 2.0) * (np.abs(u_vals)**6)
    em_base = 1.0 + act_amp * (1.0 - u_vals**2) + act_skew * u_vals + ta_boost
    
    em_R = np.maximum(0.0, em_base + act_lr * 1.0 + act_hall * u_vals * 1.0)
    em_L = np.maximum(0.0, em_base + act_lr * (-1.0) + act_hall * u_vals * (-1.0))
    
    mask = np.abs(y_vals) <= L/2
    em_R = np.where(mask, em_R, 0.0)
    em_L = np.where(mask, em_L, 0.0)

    ax_em = ax_vert.twinx()
    ax_em.fill_between(y_vals, em_R, em_L, color='red', alpha=0.1)
    ax_em.plot(y_vals, em_R, color='red', ls='-', lw=2, label='Right Leg (E x B Upward Drift)')
    ax_em.plot(y_vals, em_L, color='red', ls='--', lw=2, label='Left Leg (E x B Downward Drift)')

    ax_em.set_ylabel("Relative Target Emission", color='r')
    ax_em.set_ylim(0, max(2.0, np.max([np.max(em_R), np.max(em_L)])*1.2))
    
    rate_y = get_plot_flux(np.full_like(y_vals, x_peak), y_vals, act_interp, act_rate, base_c)
    ax_vert.plot(y_vals, rate_y, color='#1E3A8A', lw=2.5, label='Diffused Flux (at peak)')
    
    lines, labels = ax_vert.get_legend_handles_labels()
    lines2, labels2 = ax_em.get_legend_handles_labels()
    ax_vert.legend(lines + lines2, labels + labels2, loc='lower center', fontsize=8)

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

fig.text(0.5, 0.02, 'Simulation provided by AZ Thin Film Research  |  www.azthinfilm.com', 
         ha='center', va='center', fontsize=12, color='#555555', style='italic', weight='bold')

st.pyplot(fig)
