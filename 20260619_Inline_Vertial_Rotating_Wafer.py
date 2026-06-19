"""
Advanced Deposition Uniformity Simulator (Spyder Desktop Edition)
-------------------------------------------------------------------------------
* Optimized for Spyder IDE interactive plotting.
* Emojis removed to prevent Font Glyph warnings.
* Includes MaxNLocator fix to prevent garbled colorbar text at low RPMs.
"""

import matplotlib
# Attempt to automatically force an interactive pop-up window backend
try:
    matplotlib.use('Qt5Agg') 
except:
    pass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from matplotlib.widgets import Slider
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

class DepositionSimulator:
    def __init__(self):
        # --- System & Geometric Parameters ---
        self.L_initial = 800.0         
        self.d = 100.0                 
        self.rpm_initial = 60.0        
        self.v_x_initial = 10.0        
        self.amp_initial = 0.0         
        self.skew_initial = 0.0        
        
        self.R_wafer = 150.0           
        self.edge_exclusion = 3.0      
        
        self.X_start = -1000.0
        self.X_end = 1000.0
        
        self._generate_49_points()
        self._setup_grid()
        self._init_figure()
        
    def _generate_49_points(self):
        R_max = self.R_wafer - self.edge_exclusion
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
                
        self.wx, self.wy = np.array(wx), np.array(wy)

    def _setup_grid(self):
        grid_r = np.linspace(0, self.R_wafer, 40)
        grid_theta = np.linspace(0, 2*np.pi, 40)
        R_grid, T_grid = np.meshgrid(grid_r, grid_theta)
        self.grid_x = R_grid * np.cos(T_grid)
        self.grid_y = R_grid * np.sin(T_grid)

    def deposition_rate(self, x, y, L, amp, skew):
        A2 = x**2 + self.d**2
        A = np.sqrt(A2)
        
        c0, c1, c2 = 1.0 + amp, 2.0 * skew / L, -4.0 * amp / (L**2)
        K0 = c0 + c1 * y + c2 * (y**2)
        K1 = c1 + 2.0 * c2 * y
        K2 = c2
        
        v2, v1 = (L / 2.0) - y, (-L / 2.0) - y
        
        def evaluate_integral(v):
            sq = np.sqrt(A2 + v**2)
            J0 = v / (A2 * sq)
            J1 = -1.0 / sq
            J2 = np.arcsinh(v / A) - v / sq 
            return K0 * J0 + K1 * J1 + K2 * J2

        flux = self.d * (evaluate_integral(v2) - evaluate_integral(v1))
        return np.maximum(flux, 0.0)

    def compute_thickness(self, L, v_x, rpm, amp, skew):
        t_total = (self.X_end - self.X_start) / v_x
        omega = rpm * 2 * np.pi / 60.0
        revs = max((t_total * rpm / 60.0), 1)
        
        num_steps_pts = min(int(revs * 72) + 500, 25000) 
        t_pts, dt_pts = np.linspace(0, t_total, num_steps_pts, retstep=True)
        
        cx_pts = self.X_start + v_x * t_pts
        cx_pt, cos_wt, sin_wt = cx_pts[:, None], np.cos(omega * t_pts)[:, None], np.sin(omega * t_pts)[:, None]
        x_pt, y_pt = self.wx[None, :], self.wy[None, :]
        
        px = cx_pt + x_pt * cos_wt - y_pt * sin_wt
        py = x_pt * sin_wt + y_pt * cos_wt
        rate_pts = self.deposition_rate(px, py, L, amp, skew)
        thick_pts = np.sum(rate_pts, axis=0) * dt_pts
        
        num_steps_grid = min(int(revs * 36) + 200, 4000) 
        t_g, dt_g = np.linspace(0, t_total, num_steps_grid, retstep=True)
        cx_g = self.X_start + v_x * t_g
        
        cx_grid = cx_g[:, None, None]
        cos_wt_grid, sin_wt_grid = np.cos(omega * t_g)[:, None, None], np.sin(omega * t_g)[:, None, None]
        x_g, y_g = self.grid_x[None, :, :], self.grid_y[None, :, :]
        
        gx = cx_grid + x_g * cos_wt_grid - y_g * sin_wt_grid
        gy = x_g * sin_wt_grid + y_g * cos_wt_grid
        rate_grid = self.deposition_rate(gx, gy, L, amp, skew)
        thick_grid = np.sum(rate_grid, axis=0) * dt_g
        
        mean_th = np.mean(thick_pts)
        abs_dose = mean_th * 100.0  
        
        thick_pts_norm = (thick_pts / mean_th) * 100.0
        thick_grid_norm = (thick_grid / mean_th) * 100.0
        unif = (np.max(thick_pts_norm) - np.min(thick_pts_norm)) / 2.0
        
        return thick_pts_norm, thick_grid_norm, unif, abs_dose

    def _init_figure(self):
        plt.style.use('default')
        self.fig = plt.figure(figsize=(16, 9.5))
        self.fig.canvas.manager.set_window_title("Quotation Dashboard: Rotary Cathode Uniformity")
        
        self.gs = GridSpec(2, 2, figure=self.fig, left=0.28, right=0.96, top=0.94, bottom=0.06, wspace=0.25, hspace=0.35)
        
        # [Panel 1] Flux Profile
        self.ax_profile = self.fig.add_subplot(self.gs[0, 0])
        self.line_profile, = self.ax_profile.plot([], [], 'b-', lw=2.5, label='Diffused Flux at Substrate')
        self.ax_profile.axvspan(-self.R_wafer, self.R_wafer, color='gray', alpha=0.2, label='300mm Wafer Extent')
        self.ax_emission = self.ax_profile.twinx()
        self.line_emission, = self.ax_emission.plot([], [], 'r--', lw=2, label='Source Emission I(y)')
        self.ax_emission.set_ylabel("Source Emission Profile", color='r')
        self.ax_emission.tick_params(axis='y', labelcolor='r')
        
        lines, labels = self.ax_profile.get_legend_handles_labels()
        lines2, labels2 = self.ax_emission.get_legend_handles_labels()
        self.ax_profile.legend(lines + lines2, labels + labels2, loc='lower center', fontsize=9)
        
        # [Panel 2] Uniformity Map
        self.ax_map = self.fig.add_subplot(self.gs[0, 1])
        div_map = make_axes_locatable(self.ax_map)
        self.cax_map = div_map.append_axes("right", size="5%", pad=0.05)
        
        # [Panel 3] Chamber Top-Down Field
        self.ax_chamber = self.fig.add_subplot(self.gs[1, 0])
        div_chamber = make_axes_locatable(self.ax_chamber)
        self.cax_chamber = div_chamber.append_axes("right", size="5%", pad=0.05)
        
        self.cx_grid = np.linspace(-600, 600, 100)
        self.cy_grid = np.linspace(-600, 600, 100)
        self.last_chamber_params = None
        
        # [Panel 4] 49-Point Stats Scatter
        self.ax_stats = self.fig.add_subplot(self.gs[1, 1])
        self.pt_radii = np.sqrt(self.wx**2 + self.wy**2)
        self.scatter_stats = self.ax_stats.scatter([], [], c='blue', s=45, alpha=0.7, edgecolors='black')
        self.ax_stats.axhline(100, color='gray', linestyle='--', lw=2)
        self.ax_stats.set_title("SEMI Standard 49-Point Radial Analysis", fontweight='bold')
        self.ax_stats.set_xlabel("Distance from Wafer Center (mm)")
        self.ax_stats.set_ylabel("Normalized Thickness (%)")
        self.ax_stats.grid(True, linestyle='--', alpha=0.5)
        
        # --- Left Side Control Panel ---
        ui_x, ui_w, ui_h = 0.03, 0.18, 0.02
        y_start, y_step = 0.88, 0.08
        colors = ['#e8e8e8', '#e8e8e8', '#e8e8e8', '#ffe6e6', '#e6f2ff']
        
        self.fig.text(ui_x, y_start - 0*y_step + 0.025, 'Cathode Length (mm)', fontweight='bold', fontsize=10)
        self.fig.text(ui_x, y_start - 1*y_step + 0.025, 'Wafer Rotation (RPM)', fontweight='bold', fontsize=10)
        self.fig.text(ui_x, y_start - 2*y_step + 0.025, 'Translation Speed (mm/s)', fontweight='bold', fontsize=10)
        self.fig.text(ui_x, y_start - 3*y_step + 0.025, 'Profile Bow/Amp (%)  [- is Dogbone]', fontweight='bold', fontsize=10)
        self.fig.text(ui_x, y_start - 4*y_step + 0.025, 'Profile Skew (%)  [+ is Top-Heavy]', fontweight='bold', fontsize=10)
        
        self.ax_L    = self.fig.add_axes([ui_x, y_start - 0*y_step, ui_w, ui_h], facecolor=colors[0])
        self.ax_rpm  = self.fig.add_axes([ui_x, y_start - 1*y_step, ui_w, ui_h], facecolor=colors[1])
        self.ax_vx   = self.fig.add_axes([ui_x, y_start - 2*y_step, ui_w, ui_h], facecolor=colors[2])
        self.ax_amp  = self.fig.add_axes([ui_x, y_start - 3*y_step, ui_w, ui_h], facecolor=colors[3])
        self.ax_skew = self.fig.add_axes([ui_x, y_start - 4*y_step, ui_w, ui_h], facecolor=colors[4])
        
        self.s_L    = Slider(self.ax_L, '', 650.0, 900.0, valinit=self.L_initial, valstep=10)
        self.s_rpm  = Slider(self.ax_rpm, '', 0.0, 120.0, valinit=self.rpm_initial, valstep=5)
        self.s_vx   = Slider(self.ax_vx, '', 1.0, 30.0, valinit=self.v_x_initial, valstep=1)
        self.s_amp  = Slider(self.ax_amp, '', -50.0, 50.0, valinit=self.amp_initial, valstep=1)
        self.s_skew = Slider(self.ax_skew, '', -50.0, 50.0, valinit=self.skew_initial, valstep=1)
        
        self.ax_text = self.fig.add_axes([ui_x, 0.04, ui_w + 0.02, 0.44])
        self.ax_text.axis('off')
        self.result_text = self.ax_text.text(0.02, 0.95, "", fontsize=12, family='monospace', va='top', ha='left',
                                             bbox=dict(facecolor='white', alpha=0.95, edgecolor='gray', lw=2, boxstyle='round,pad=1.2'))
        
        for slider in [self.s_L, self.s_rpm, self.s_vx, self.s_amp, self.s_skew]:
            slider.on_changed(self.update)
            
        self.update(None)
        plt.show()

    def update(self, val):
        L, rpm, v_x = self.s_L.val, self.s_rpm.val, self.s_vx.val
        amp, skew = self.s_amp.val / 100.0, self.s_skew.val / 100.0
        
        thick_pts_norm, thick_grid_norm, unif, abs_dose = self.compute_thickness(L, v_x, rpm, amp, skew)
        dev = max(0.5, unif * 1.5) 
        
        # 1. Update Profile Curves
        y_vals = np.linspace(-600, 600, 200)
        u_vals = 2 * y_vals / L
        mask = (y_vals >= -L/2) & (y_vals <= L/2)
        emission = np.zeros_like(y_vals)
        emission[mask] = 1.0 + amp * (1.0 - u_vals[mask]**2) + skew * u_vals[mask]
        
        self.line_emission.set_data(y_vals, emission)
        self.ax_emission.set_ylim(0, max(2.0, np.max(emission)*1.2))
        
        rate_y = self.deposition_rate(0, y_vals, L, amp, skew)
        self.line_profile.set_data(y_vals, rate_y)
        self.ax_profile.set_xlim(-600, 600)
        self.ax_profile.set_ylim(0, max(0.001, np.max(rate_y)*1.1))
        self.ax_profile.set_title("Cathode Source & Diffused Flux", fontweight='bold')
        self.ax_profile.set_xlabel("Cathode Axis Y (mm)")
        self.ax_profile.set_ylabel("Relative Flux Intensity", color='b')
        
        # 2. Update Wafer Map Contour
        self.ax_map.clear()
        self.cax_map.clear()
        self.ax_map.set_title("Wafer Uniformity Map (Mean = 100%)", fontweight='bold')
        self.ax_map.set_xlabel("Wafer X (mm)")
        self.ax_map.set_ylabel("Wafer Y (mm)")
        self.ax_map.set_aspect('equal')
        self.ax_map.add_patch(Circle((0,0), self.R_wafer, fill=False, color='black', lw=2))
        
        levels = np.linspace(100 - dev, 100 + dev, 40)
        contour = self.ax_map.contourf(self.grid_x, self.grid_y, thick_grid_norm, levels=levels, cmap='viridis', extend='both')
        self.ax_map.scatter(self.wx, self.wy, c='white', s=15, edgecolors='black', zorder=5)
        
        # Application of MaxNLocator to prevent text garbling on the desktop app
        cbar_map = self.fig.colorbar(contour, cax=self.cax_map)
        cbar_map.set_label("Normalized Thickness (%)")
        cbar_map.locator = MaxNLocator(nbins=6)
        cbar_map.update_ticks()
        
        # 3. Update Chamber Environment Contour
        current_chamber_params = (L, amp, skew)
        if current_chamber_params != self.last_chamber_params:
            self.ax_chamber.clear()
            self.cax_chamber.clear()
            self.ax_chamber.set_title("Chamber Simulation Environment", fontweight='bold')
            self.ax_chamber.set_xlabel("Translation Axis X (mm)")
            self.ax_chamber.set_ylabel("Cathode Axis Y (mm)")
            self.ax_chamber.set_aspect('equal')
            
            CX, CY = np.meshgrid(self.cx_grid, self.cy_grid)
            rate_2d = self.deposition_rate(CX, CY, L, amp, skew)
            chamber_contour = self.ax_chamber.contourf(CX, CY, rate_2d, levels=40, cmap='magma')
            
            cbar_chamber = self.fig.colorbar(chamber_contour, cax=self.cax_chamber)
            cbar_chamber.set_label("Relative Source Flux")
            cbar_chamber.locator = MaxNLocator(nbins=6)
            cbar_chamber.update_ticks()
            
            self.ax_chamber.plot([0, 0], [-L/2, L/2], 'w-', lw=5, solid_capstyle='round', label="Target Width")
            self.ax_chamber.axhline(0, color='cyan', linestyle='--', alpha=0.8, lw=2, label="Wafer Center Path")
            self.ax_chamber.legend(loc='upper right')
            self.last_chamber_params = current_chamber_params
            
        # 4. Update 49-Point Scatter Stats
        self.scatter_stats.set_offsets(np.c_[self.pt_radii, thick_pts_norm])
        self.ax_stats.set_xlim(-5, 155)
        self.ax_stats.set_ylim(100 - dev, 100 + dev)
        
        # 5. Update Evaluation Text Box (Emojis Removed)
        goal_met = unif <= 1.0
        color_eval = "forestgreen" if goal_met else "firebrick"
        
        stats_text = (
            f"=== SYSTEM SETTINGS ===\n"
            f"Target Length:     {L:4.0f} mm\n"
            f"Wafer Rotation:    {rpm:4.0f} RPM\n"
            f"Translation Spd:   {v_x:4.0f} mm/s\n"
            f"Profile Bow/Amp:   {amp*100:+4.0f} %\n"
            f"Profile Skew:      {skew*100:+4.0f} %\n\n"
            f"=== 49-POINT WAFER MAP ===\n"
            f"Absolute Dose:     {abs_dose:6.1f} a.u.\n"
            f"Mean Thickness:    100.00 %\n"
            f"Max Thickness:     {np.max(thick_pts_norm):6.2f} %\n"
            f"Min Thickness:     {np.min(thick_pts_norm):6.2f} %\n"
            f"Uniformity (+/-):  {unif:6.2f} %\n\n"
            f"TARGET (≤ ±1.0%):  {'[PASS] ACHIEVED' if goal_met else '[FAIL] NOT MET'}"
        )
        self.result_text.set_text(stats_text)
        self.result_text.get_bbox_patch().set_edgecolor(color_eval)
        self.result_text.get_bbox_patch().set_facecolor("#f0fff0" if goal_met else "#fff0f0")
        
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    app = DepositionSimulator()
