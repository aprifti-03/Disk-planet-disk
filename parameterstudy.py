import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from scipy.integrate import solve_ivp
from datetime import datetime
import os

# Constants
G = 4*np.pi**2

@dataclass
class SystemParameters:
    """Container for system parameters"""
    M_star: float
    M_c: float
    M_cs: float
    M_cb: float
    a_c: float
    R_cs: float
    R_in: float
    R_out: float
    init_theta_cs_c: float
    init_theta_cs_cb: float
    init_theta_c_cb: float

class SecularResonanceSystem:
    """
    Implements the secular precession resonance model from Owen & Lai (2017)
    """

    def __init__(self, params: SystemParameters):
        self.params = params

        # Initialize unit vectors (L_hat)
        self.l_cs = np.array([0.0, 0.0, 1.0])
        self.l_c = np.array([0.0, -np.sin(params.init_theta_cs_c), np.cos(params.init_theta_cs_c)])
        self.l_cb = np.array([-np.sin(params.init_theta_cs_cb), 0.0, np.cos(params.init_theta_cs_cb)])

        # Normalize
        self.l_cs = self.l_cs / np.linalg.norm(self.l_cs)
        self.l_c = self.l_c / np.linalg.norm(self.l_c)
        self.l_cb = self.l_cb / np.linalg.norm(self.l_cb)

        # Initialize precession frequency attributes
        self.Omega_cs_c = 0.0
        self.Omega_cs_cb = 0.0
        self.Omega_c_cb = 0.0
        self.Omega_c_cs = 0.0
        self.Omega_cb_c = 0.0
        self.Omega_cb_cs = 0.0
        self.Omega_cs = 0.0
        self.Omega_c = 0.0
        self.Omega_cb = 0.0

        # Calculate initial angular momentum magnitudes
        self._update_angular_momenta()

        # Evolution parameters
        self.t_viscous = 2e5
        self.f_accrete = 0.1
        self.migration_timescale = None
        self.scenario = 'accreting'

        # Store initial parameters
        self.M_cs_initial = params.M_cs
        self.M_cb_initial = params.M_cb
        self.M_c_initial = params.M_c
        self.a_c_initial = params.a_c

        # Store current time
        self.current_time = 0.0

        # History tracking
        self.history = {
            'time': [], 'M_cs': [], 'M_c': [], 'M_cb': [],
            'R_cs': [], 'R_in': [], 'a_c': [],
            'l_cs_x': [], 'l_cs_y': [], 'l_cs_z': [],
            'l_c_x': [], 'l_c_y': [], 'l_c_z': [],
            'l_cb_x': [], 'l_cb_y': [], 'l_cb_z': [],
            'theta_cs_c': [], 'theta_cs_cb': [], 'theta_c_cb': [],
            'Omega_cs': [], 'Omega_c': [], 'Omega_cb': [],
            'Omega_cs_c': [], 'Omega_c_cb': [], 'Omega_cs_cb': [],
            'Omega_r': [], 'theta_Lr_cb': [], 'L_r_x': [], 'L_r_y': [], 'L_r_z': []
        }

    def _update_angular_momenta(self):
        """Calculate angular momentum magnitudes"""
        self.L_cs_mag = (2/3) * self.params.M_cs * np.sqrt(G * self.params.M_star * self.params.R_cs)
        self.L_c_mag = self.params.M_c * np.sqrt(G * self.params.M_star * self.params.a_c)
        self.L_cb_mag = (2/3) * self.params.M_cb * np.sqrt(G * self.params.M_star * self.params.R_out)

    def _update_precession_frequencies(self, t=None):
        """Calculate precession frequencies from equation (16)"""
        Omega_K_cs = np.sqrt(G * self.params.M_star / self.params.R_cs**3)
        Omega_K_ac = np.sqrt(G * self.params.M_star / self.params.a_c**3)
        Omega_K_out = np.sqrt(G * self.params.M_star / self.params.R_out**3)

        dot_cs_c = np.dot(self.l_cs, self.l_c)
        dot_cs_cb = np.dot(self.l_cs, self.l_cb)
        dot_c_cb = np.dot(self.l_c, self.l_cb)

        self.Omega_cs_c = (-3/8) * Omega_K_cs * (self.params.M_c / self.params.M_star) * \
                          (self.params.R_cs / self.params.a_c)**3 * dot_cs_c

        self.Omega_cs_cb = (-3/16) * Omega_K_cs * (self.params.M_cb / self.params.M_star) * \
                           (self.params.R_cs**3 / (self.params.R_out * self.params.R_in**2)) * dot_cs_cb

        self.Omega_c_cb = (-3/8) * Omega_K_ac * (self.params.M_cb / self.params.M_star) * \
                          (self.params.a_c**3 / (self.params.R_out * self.params.R_in**2)) * dot_c_cb

        self.Omega_c_cs = (-1/4) * Omega_K_ac * (self.params.M_cs / self.params.M_star) * \
                          (self.params.R_cs / self.params.a_c)**2 * dot_cs_c

        self.Omega_cb_c = (-3/4) * Omega_K_out * (self.params.M_c / self.params.M_star) * \
                          (self.params.a_c / self.params.R_in)**2 * dot_c_cb

        self.Omega_cb_cs = (-3/16) * Omega_K_out * (self.params.M_cs / self.params.M_star) * \
                           (self.params.R_cs / self.params.R_in)**2 * dot_cs_cb

        self.Omega_cs = self.Omega_cs_c + self.Omega_cs_cb
        self.Omega_c = self.Omega_c_cb + self.Omega_c_cs
        self.Omega_cb = self.Omega_cb_c + self.Omega_cb_cs

    def _compute_dl_dt(self, t, state):
        """Compute the rate of change of unit vectors"""
        l_cs = state[0:3]
        l_c = state[3:6]
        l_cb = state[6:9]

        self.l_cs = l_cs
        self.l_c = l_c
        self.l_cb = l_cb

        self._update_parameters_with_time(t)
        self._update_precession_frequencies(t)

        Omega_cs_vector = self.Omega_cs_c * l_c + self.Omega_cs_cb * l_cb
        dl_cs_dt = np.cross(Omega_cs_vector, l_cs)

        Omega_c_vector = self.Omega_c_cs * l_cs + self.Omega_c_cb * l_cb
        dl_c_dt = np.cross(Omega_c_vector, l_c)

        Omega_cb_vector = self.Omega_cb_cs * l_cs + self.Omega_cb_c * l_c
        dl_cb_dt = np.cross(Omega_cb_vector, l_cb)

        return np.concatenate([dl_cs_dt, dl_c_dt, dl_cb_dt])

    def _update_parameters_with_time(self, t):
        """Update masses and radii based on current time"""
        if self.scenario == 'accreting':
            self._update_masses_and_radii_accreting_time(t)
        elif self.scenario == 'migrating':
            self._update_masses_and_radii_migrating_time(t)

    def _update_masses_and_radii_accreting_time(self, t):
        """Update masses and radii for accreting companion scenario"""
        t_factor = np.sqrt(1 + t / self.t_viscous)

        M_cb_new = self.M_cb_initial / t_factor
        M_cs_new = self.M_cs_initial / t_factor**3

        depleted_mass = self.M_cb_initial - M_cb_new
        M_c_new = self.M_c_initial + self.f_accrete * depleted_mass

        self.params.M_cb = M_cb_new
        self.params.M_cs = M_cs_new
        self.params.M_c = M_c_new

        # Update truncation radii based on current a_c
        self._update_truncation_radii()

        self._update_angular_momenta()

    def _update_masses_and_radii_migrating_time(self, t):
        """Update masses and radii for migrating companion scenario"""
        if self.migration_timescale is None:
            return

        a_c_new = self.a_c_initial * np.exp(-t / self.migration_timescale)
        self.params.a_c = max(0.5, a_c_new)

        self._update_truncation_radii()

        dM_cs = -self.params.M_cs * (1 - np.exp(-t / self.migration_timescale))
        dM_cb = -dM_cs

        self.params.M_cs = self.M_cs_initial + dM_cs
        self.params.M_cb = self.M_cb_initial + dM_cb

        self.params.M_cs = max(0, self.params.M_cs)
        self.params.M_cb = max(0, self.params.M_cb)

        self._update_angular_momenta()

    def _update_truncation_radii(self):
        """Update R_cs and R_in based on current a_c and M_c (equations 41-42)"""
        q_ratio = self.params.M_c / (3 * self.params.M_star)
        if q_ratio > 0 and q_ratio < 1:
            self.params.R_cs = self.params.a_c * (1 - 3 * q_ratio**(1/3))
            self.params.R_in = self.params.a_c * (1 + 3 * q_ratio**(1/3))

        # Ensure radii are physical
        self.params.R_cs = max(0.1, min(self.params.R_cs, self.params.a_c * 0.9))
        self.params.R_in = max(self.params.R_cs + 0.5, self.params.R_in)

    def analyze_rotating_frame_dynamics(self):
        """Calculate rotating frame quantities from Section 3.1"""
        self._update_precession_frequencies()

        Omega_r_Lr = self.Omega_cs_c * self.l_c - self.Omega_c_cb * self.l_cb
        Omega_r = np.linalg.norm(Omega_r_Lr)

        if Omega_r > 0:
            L_r = Omega_r_Lr / Omega_r
            theta_Lr_cb = np.degrees(np.arccos(np.clip(np.dot(L_r, self.l_cb), -1, 1)))
        else:
            L_r = np.array([0, 0, 0])
            theta_Lr_cb = 90.0

        return {
            'Omega_r': Omega_r,
            'theta_Lr_cb': theta_Lr_cb,
            'L_r': L_r
        }

    def _record_state(self):
        """Record current state"""
        self.history['time'].append(self.current_time)
        self.history['M_cs'].append(self.params.M_cs)
        self.history['M_c'].append(self.params.M_c)
        self.history['M_cb'].append(self.params.M_cb)
        self.history['R_cs'].append(self.params.R_cs)
        self.history['R_in'].append(self.params.R_in)
        self.history['a_c'].append(self.params.a_c)

        self.history['l_cs_x'].append(self.l_cs[0])
        self.history['l_cs_y'].append(self.l_cs[1])
        self.history['l_cs_z'].append(self.l_cs[2])
        self.history['l_c_x'].append(self.l_c[0])
        self.history['l_c_y'].append(self.l_c[1])
        self.history['l_c_z'].append(self.l_c[2])
        self.history['l_cb_x'].append(self.l_cb[0])
        self.history['l_cb_y'].append(self.l_cb[1])
        self.history['l_cb_z'].append(self.l_cb[2])

        theta_cs_c = np.degrees(np.arccos(np.clip(np.dot(self.l_cs, self.l_c), -1, 1)))
        theta_cs_cb = np.degrees(np.arccos(np.clip(np.dot(self.l_cs, self.l_cb), -1, 1)))
        theta_c_cb = np.degrees(np.arccos(np.clip(np.dot(self.l_c, self.l_cb), -1, 1)))

        self.history['theta_cs_c'].append(theta_cs_c)
        self.history['theta_cs_cb'].append(theta_cs_cb)
        self.history['theta_c_cb'].append(theta_c_cb)

        self.history['Omega_cs'].append(self.Omega_cs)
        self.history['Omega_c'].append(self.Omega_c)
        self.history['Omega_cb'].append(self.Omega_cb)
        self.history['Omega_cs_c'].append(self.Omega_cs_c)
        self.history['Omega_c_cb'].append(self.Omega_c_cb)
        self.history['Omega_cs_cb'].append(self.Omega_cs_cb)

        rot_frame = self.analyze_rotating_frame_dynamics()
        self.history['Omega_r'].append(rot_frame['Omega_r'])
        self.history['theta_Lr_cb'].append(rot_frame['theta_Lr_cb'])
        self.history['L_r_x'].append(rot_frame['L_r'][0])
        self.history['L_r_y'].append(rot_frame['L_r'][1])
        self.history['L_r_z'].append(rot_frame['L_r'][2])

    def evolve(self, total_time: float, scenario: str = 'accreting',
               migration_timescale: float = None, f_accrete: float = 0.1,
               t_viscous: float = 2e5, max_step: float = 5000.0,
               rtol: float = 1e-8, atol: float = 1e-10, verbose: bool = False):
        """Evolve the system using solve_ivp"""
        self.scenario = scenario
        self.f_accrete = f_accrete
        self.t_viscous = t_viscous
        self.migration_timescale = migration_timescale

        state0 = np.concatenate([self.l_cs, self.l_c, self.l_cb])

        def ode_func(t, y):
            return self._compute_dl_dt(t, y)

        self.current_time = 0.0
        self._update_precession_frequencies(0.0)
        self._record_state()

        solution = solve_ivp(
            ode_func, [0, total_time], state0,
            method='DOP853', rtol=rtol, atol=atol,
            max_step=max_step, dense_output=True
        )

        if not solution.success and verbose:
            print(f"Warning: Integration failed: {solution.message}")

        times = solution.t
        states = solution.y.T

        for i, (t, state) in enumerate(zip(times, states)):
            self.current_time = t
            self.l_cs = state[0:3]
            self.l_c = state[3:6]
            self.l_cb = state[6:9]

            norm_cs = np.linalg.norm(self.l_cs)
            norm_c = np.linalg.norm(self.l_c)
            norm_cb = np.linalg.norm(self.l_cb)

            if norm_cs > 0:
                self.l_cs /= norm_cs
            if norm_c > 0:
                self.l_c /= norm_c
            if norm_cb > 0:
                self.l_cb /= norm_cb

            self._update_parameters_with_time(t)
            self._update_precession_frequencies(t)
            self._record_state()

    def get_final_state(self):
        """Return the final state of the simulation"""
        if len(self.history['time']) == 0:
            return {}

        return {
            'final_M_cs': self.history['M_cs'][-1],
            'final_M_c': self.history['M_c'][-1],
            'final_M_cb': self.history['M_cb'][-1],
            'final_a_c': self.history['a_c'][-1],
            'final_R_cs': self.history['R_cs'][-1],
            'final_R_in': self.history['R_in'][-1],
            'final_theta_cs_c': self.history['theta_cs_c'][-1],
            'final_theta_cs_cb': self.history['theta_cs_cb'][-1],
            'final_theta_c_cb': self.history['theta_c_cb'][-1],
            'max_theta_cs_cb': max(self.history['theta_cs_cb']),
            'final_Omega_cs': self.history['Omega_cs'][-1],
            'final_Omega_c': self.history['Omega_c'][-1],
        }


# ---------------------------------------------------------------------------
# Limit-line helper (unchanged physics, but now called on pcolormesh axes)
# ---------------------------------------------------------------------------

def add_simulation_overlays(ax, M_cb_values, M_c_ratios=None,
                            f_accrete_values=None):
    """
    Add overlays to a heatmap plotted in the TRUE (M_c/M_star, M_cb/M_star)
    plane.

    The only data-driven overlay is the blue "Simulation max" line, which
    traces the rightmost sampled M_c for each M_cb row (i.e. the M_c value at
    the largest f_accrete in that row). The f-value annotations at the right
    edge simply label which f_accrete column sits there; they are NOT a
    physical bound.
    """
    M_cb_values = np.asarray(M_cb_values)

    # Actual simulation maximum M_c for each M_cb row (right edge of data)
    if M_c_ratios is not None:
        with np.errstate(all='ignore'):
            max_M_c_for_each_Mcb = np.nanmax(M_c_ratios, axis=1)
        ax.plot(max_M_c_for_each_Mcb, M_cb_values, 'b--', linewidth=1.5,
                alpha=0.7, label='Simulation max')

    # Annotate the right edge with the f_accrete value that sits there
    if f_accrete_values is not None and M_c_ratios is not None:
        # For each row, find which column attains the max M_c
        # (should be the last column, f = max, but be robust)
        with np.errstate(all='ignore'):
            col_idx = np.nanargmax(M_c_ratios, axis=1)
        y_pos = M_cb_values[-1] * 0.95
        # Use the last row's column index (should be the same for all rows)
        j_last = int(col_idx[-1]) if len(col_idx) else len(f_accrete_values) - 1
        f_last = f_accrete_values[j_last]
        x_last = np.nanmax(M_c_ratios[-1, :])
        ax.annotate(f'f={f_last:.1f}', xy=(x_last, y_pos),
                    xytext=(0, -5), textcoords='offset points',
                    fontsize=7, alpha=0.6, ha='center')

# ---------------------------------------------------------------------------
# Study driver
# ---------------------------------------------------------------------------

def run_heatmap_study(a_c_ratios: List[float], M_cb_values: List[float],
                      f_accrete_values: List[float], M_star: float = 2,
                      M_cs_initial: float = 0.003, R_out: float = 300.0,
                      total_time: float = 3e6, t_viscous: float = 2e5,
                      initial_misalignment: float = 5.0, verbose: bool = True):
    """
    Run a double loop study to create heatmaps of misalignment angles.
    """

    # Create results directory
    os.makedirs('heatmap_results16', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Store results for each a_c/R_out ratio
    all_results = {}

    print("=" * 80)
    print("HEATMAP STUDY: Misalignment as function of M_c/M_star and M_cb/M_star")
    print(f"M_star = {M_star}, M_cs_initial = {M_cs_initial}, R_out = {R_out}")
    print(f"f_accrete values: {f_accrete_values}")
    print(f"a_c/R_out ratios: {a_c_ratios}")
    print("=" * 80)

    for a_ratio_idx, a_c_ratio in enumerate(a_c_ratios):
        a_c = a_c_ratio * R_out

        print(f"\n[{a_ratio_idx+1}/{len(a_c_ratios)}] Processing a_c/R_out = {a_c_ratio:.3f} (a_c = {a_c:.1f} AU)")

        # Initialize arrays for this a_c ratio
        n_M_cb = len(M_cb_values)
        n_f = len(f_accrete_values)
        M_c_ratios = np.zeros((n_M_cb, n_f))
        theta_final = np.zeros((n_M_cb, n_f))
        theta_max = np.zeros((n_M_cb, n_f))
        theta_excitation = np.zeros((n_M_cb, n_f))

        # Determine R_cs and R_in based on initial conditions
        R_cs_initial = a_c * 0.94
        R_in_initial = a_c * 1.06

        for i, M_cb_ratio in enumerate(M_cb_values):
            M_cb = M_cb_ratio * M_star

            for j, f_accrete in enumerate(f_accrete_values):
                M_c_initial = 0.001

                params = SystemParameters(
                    M_star=M_star,
                    M_c=M_c_initial,
                    M_cs=M_cs_initial,
                    M_cb=M_cb,
                    a_c=a_c,
                    R_cs=R_cs_initial,
                    R_in=R_in_initial,
                    R_out=R_out,
                    init_theta_cs_c=np.radians(initial_misalignment),
                    init_theta_cs_cb=np.radians(initial_misalignment),
                    init_theta_c_cb=np.radians(initial_misalignment)
                )

                system = SecularResonanceSystem(params)

                try:
                    system.evolve(total_time=total_time, scenario='accreting',
                                  f_accrete=f_accrete, t_viscous=t_viscous,
                                  max_step=5000.0, verbose=False)

                    final_state = system.get_final_state()

                    M_c_final = final_state['final_M_c']
                    M_cb_final = final_state['final_M_cb']

                    M_c_ratios[i, j] = M_c_final / M_star
                    theta_final[i, j] = final_state['final_theta_cs_cb']
                    theta_max[i, j] = final_state['max_theta_cs_cb']
                    theta_excitation[i, j] = final_state['max_theta_cs_cb'] - initial_misalignment

                    if verbose and (i * len(f_accrete_values) + j) % 100 == 0:
                        print(f"  M_cb={M_cb_ratio:.3f}, f={f_accrete:.2f}: "
                              f"M_c/M*={M_c_ratios[i,j]:.4f}, θ={theta_final[i,j]:.1f}°")

                except Exception as e:
                    if verbose:
                        print(f"  Failed for M_cb={M_cb_ratio:.3f}, f={f_accrete:.2f}: {str(e)[:50]}...")
                    M_c_ratios[i, j] = np.nan
                    theta_final[i, j] = np.nan
                    theta_max[i, j] = np.nan
                    theta_excitation[i, j] = np.nan

        all_results[a_c_ratio] = {
            'a_c': a_c,
            'M_c_ratios': M_c_ratios,
            'theta_final': theta_final,
            'theta_max': theta_max,
            'theta_excitation': theta_excitation,
            'M_cb_values': M_cb_values,
            'f_accrete_values': f_accrete_values
        }

        plot_heatmap(all_results[a_c_ratio], a_c_ratio, timestamp)

    save_heatmap_results(all_results, timestamp)
    plot_combined_heatmaps(all_results, timestamp)

    return all_results


# ---------------------------------------------------------------------------
# CORRECTED plotting: pcolormesh with true (M_c, M_cb) coordinates
# ---------------------------------------------------------------------------

def _build_coordinate_grids(result: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the true 2D coordinate grids from the result dictionary.

    Returns
    -------
    M_c_grid : ndarray, shape (n_Mcb, n_f)
        True final M_c/M_star for each cell.
    M_cb_grid : ndarray, shape (n_Mcb, n_f)
        True M_cb/M_star for each cell (constant along rows).
    """
    M_c_ratios = result['M_c_ratios']            # shape (n_Mcb, n_f)
    M_cb_values = np.asarray(result['M_cb_values'])
    n_M_cb, n_f = M_c_ratios.shape

    # M_cb is constant along each row (indexed by i)
    M_cb_grid = np.repeat(M_cb_values[:, None], n_f, axis=1)

    # M_c_ratios already gives the true M_c/M_star of each cell
    M_c_grid = M_c_ratios.copy()

    return M_c_grid, M_cb_grid


def plot_heatmap(result: Dict, a_c_ratio: float, timestamp: str):
    """
    Plot a single heatmap for a given a_c/R_out ratio.

    Uses pcolormesh with the TRUE 2D (M_c/M_star, M_cb/M_star) coordinates of
    each simulation cell, so that the x-axis really is M_c/M_star and the
    overlaid physical-limit lines align with the data.
    """
    M_c_ratios = result['M_c_ratios']
    theta_final = result['theta_final']
    theta_max = result['theta_max']
    theta_excitation = result['theta_excitation']
    M_cb_values = np.asarray(result['M_cb_values'])
    f_accrete_values = result['f_accrete_values']
    a_c = result['a_c']

    # Build the true coordinate grids
    M_c_grid, M_cb_grid = _build_coordinate_grids(result)

    # Common color scale limits for the two misalignment panels
    theta_final_plot = np.ma.masked_invalid(theta_final)
    theta_max_plot = np.ma.masked_invalid(theta_max)
    vmin = min(np.nanmin(theta_final_plot), np.nanmin(theta_max_plot))
    vmax = max(np.nanmax(theta_final_plot), np.nanmax(theta_max_plot))

    # ---- Figure 1: final + max misalignment -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 0: Final misalignment
    pcm1 = axes[0].pcolormesh(
        M_c_grid, M_cb_grid, theta_final_plot,
        shading='auto', cmap='viridis', vmin=vmin, vmax=vmax
    )
    axes[0].set_xlabel(r'$M_c$ / $M_\star$', fontsize=12)
    axes[0].set_ylabel(r'$M_{cb}$ / $M_\star$', fontsize=12)
    axes[0].set_xlim(np.nanmin(M_c_grid), np.nanmax(M_c_grid))
    axes[0].set_title(fr'Final $\theta_{{cs,cb}}$ ($a_c/R_{{out}}$ = {a_c_ratio:.2f})', fontsize=12)
    plt.colorbar(pcm1, ax=axes[0], label='Misalignment (degrees)')

    add_simulation_overlays(axes[0], M_cb_values,
                             M_c_ratios=M_c_ratios, f_accrete_values=f_accrete_values
                             )
    axes[0].legend(loc='upper left', fontsize=8)

    # Panel 1: Maximum misalignment
    pcm2 = axes[1].pcolormesh(
        M_c_grid, M_cb_grid, theta_max_plot,
        shading='auto', cmap='plasma', vmin=vmin, vmax=vmax
    )
    axes[1].set_xlabel(r'$M_c$ / $M_\star$', fontsize=12)
    axes[1].set_ylabel(r'$M_{cb}$ / $M_\star$', fontsize=12)
    axes[1].set_xlim(np.nanmin(M_c_grid), np.nanmax(M_c_grid))
    axes[1].set_title(fr'Max $\theta_{{cs,cb}}$ ($a_c/R_{{out}}$ = {a_c_ratio:.2f})', fontsize=12)
    plt.colorbar(pcm2, ax=axes[1], label='Misalignment (degrees)')

    add_simulation_overlays(axes[1], M_cb_values,
                             M_c_ratios=M_c_ratios, f_accrete_values=f_accrete_values
                             )
    axes[1].legend(loc='upper left', fontsize=8)

    plt.tight_layout()

    filename = f'heatmap_results16/heatmap_a_c_ratio_{a_c_ratio:.3f}_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap with limit lines saved: {filename}")

    # ---- Figure 2: excitation --------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.pcolormesh(
        M_c_grid, M_cb_grid, np.ma.masked_invalid(theta_excitation),
        shading='auto', cmap='RdBu_r'
    )
    ax.set_xlabel(r'$M_c$ / $M_\star$', fontsize=12)
    ax.set_ylabel(r'$M_{cb}$ / $M_\star$', fontsize=12)
    ax.set_title(fr'Misalignment Excitation ($a_c/R_{{out}}$ = {a_c_ratio:.2f})', fontsize=12)
    plt.colorbar(pcm, ax=ax, label='Excited Misalignment (degrees)')

    add_simulation_overlays(ax, M_cb_values,
                             M_c_ratios=M_c_ratios, f_accrete_values=f_accrete_values
                             )
    ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()

    filename = f'heatmap_results16/excitation_a_c_ratio_{a_c_ratio:.3f}_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_combined_heatmaps(all_results: Dict, timestamp: str):
    """
    Create a combined figure with heatmaps for all a_c/R_out ratios.

    Each panel uses pcolormesh with the panel's own true (M_c, M_cb) grid, so
    different panels can have different x-ranges without distortion.
    """
    n_ratios = len(all_results)
    if n_ratios == 0:
        return

    n_cols = min(3, n_ratios)
    n_rows = (n_ratios + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    axes_flat = axes.flatten() if n_rows > 1 else axes

    # Determine a common color scale across all panels
    all_theta = np.concatenate([
        np.asarray(result['theta_final']).ravel()
        for result in all_results.values()
    ])
    all_theta = all_theta[~np.isnan(all_theta)]
    vmin = all_theta.min() if all_theta.size else 0.0
    vmax = all_theta.max() if all_theta.size else 1.0

    last_pcm = None
    for idx, (a_c_ratio, result) in enumerate(sorted(all_results.items())):
        ax = axes_flat[idx]

        theta_final = result['theta_final']
        M_c_ratios = result['M_c_ratios']
        M_cb_values = np.asarray(result['M_cb_values'])
        f_accrete_values = result['f_accrete_values']

        # Build true coordinate grids for this panel
        M_c_grid, M_cb_grid = _build_coordinate_grids(result)

        last_pcm = ax.pcolormesh(
            M_c_grid, M_cb_grid, np.ma.masked_invalid(theta_final),
            shading='auto', cmap='viridis', vmin=vmin, vmax=vmax
        )
        ax.set_xlabel(r'$M_c$ / $M_\star$', fontsize=10)
        ax.set_ylabel(r'$M_{cb}$ / $M_\star$', fontsize=10)
        ax.set_xlim(np.nanmin(M_c_grid), np.nanmax(M_c_grid))
        ax.set_title(fr'$a_c/R_{{out}}$ = {a_c_ratio:.2f}', fontsize=11)

        add_simulation_overlays(ax, M_cb_values,
                                M_c_ratios=M_c_ratios, f_accrete_values=f_accrete_values
                                 )

    # Hide empty subplots
    for idx in range(len(all_results), len(axes_flat)):
        axes_flat[idx].axis('off')

    # One shared colorbar
    if last_pcm is not None:
        fig.colorbar(last_pcm, ax=axes_flat.tolist(),
                     label=r'$\theta_{cs,cb}$ (°)', shrink=0.8)

    plt.suptitle(r'Final Misalignment vs $M_c$/$M_\star$ and $M_{cb}$/$M_\star$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    filename = f'heatmap_results16/combined_heatmaps_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Combined heatmaps with limit lines saved: {filename}")


def save_heatmap_results(all_results: Dict, timestamp: str):
    """
    Save heatmap results to a text file
    """
    filename = f'heatmap_results16/heatmap_data_{timestamp}.txt'

    with open(filename, 'w') as f:
        f.write("# Heatmap Study Results\n")
        f.write(f"# Timestamp: {timestamp}\n")
        f.write("# Format: a_c_ratio, M_cb_ratio, f_accrete, M_c_final_ratio, theta_final, theta_max\n")

        for a_c_ratio, result in sorted(all_results.items()):
            M_c_ratios = result['M_c_ratios']
            theta_final = result['theta_final']
            theta_max = result['theta_max']
            M_cb_values = result['M_cb_values']
            f_accrete_values = result['f_accrete_values']

            for i, M_cb in enumerate(M_cb_values):
                for j, f_accrete in enumerate(f_accrete_values):
                    if not np.isnan(M_c_ratios[i, j]):
                        f.write(f"{a_c_ratio:.6f}, {M_cb:.6f}, {f_accrete:.6f}, "
                                f"{M_c_ratios[i,j]:.6f}, {theta_final[i,j]:.6f}, {theta_max[i,j]:.6f}\n")

    print(f"Data saved to: {filename}")


def main():
    """
    Main function to run the heatmap study
    """
    print("=" * 80)
    print("Secular Precession Resonance Model - Heatmap Study with Limit Lines")
    print("Owen & Lai (2017) MNRAS, 469, 2834")
    print("=" * 80)

    a_c_ratios = [0.02, 0.04, 0.08, 0.1, 0.14, 0.18 ,0.2, 0.25]

    M_cb_values = np.linspace(0.01, 0.20, 20)
    f_accrete_values = np.linspace(0.0, 1, 20)

    print(f"\nParameter ranges:")
    print(f"  a_c/R_out: {a_c_ratios}")
    print(f"  M_cb/M_star: {M_cb_values[0]:.3f} - {M_cb_values[-1]:.3f} ({len(M_cb_values)} values)")
    print(f"  f_accrete: {f_accrete_values[0]:.3f} - {f_accrete_values[-1]:.3f} ({len(f_accrete_values)} values)")
    print(f"\nTotal simulations: {len(a_c_ratios)} × {len(M_cb_values)} × {len(f_accrete_values)} = {len(a_c_ratios) * len(M_cb_values) * len(f_accrete_values)}")
    print("=" * 80)

    results = run_heatmap_study(
        a_c_ratios=a_c_ratios,
        M_cb_values=M_cb_values,
        f_accrete_values=f_accrete_values,
        M_star=2,
        M_cs_initial=0.003,
        R_out=300.0,
        total_time=3e6,
        t_viscous=2e5,
        initial_misalignment=5.0,
        verbose=True
    )

    print("\n" + "=" * 80)
    print("HEATMAP STUDY WITH LIMIT LINES COMPLETE!")
    print(f"Results saved in 'heatmap_results16' directory")
    print("=" * 80)


if __name__ == "__main__":
    main()