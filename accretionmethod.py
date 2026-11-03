import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# Constants
G = 4 * np.pi**2   # AU^3 / (M_sun * yr^2)


@dataclass
class SystemParameters:
    """Container for system parameters"""
    M_star: float              # Primary star mass (M_sun)
    M_c: float                 # Companion mass (M_sun)
    M_cs: float                # Circumstellar disc mass (M_sun)
    M_cb: float                # Circumbinary disc mass (M_sun)
    a_c: float                 # Companion orbital separation (AU)
    R_cs: float                # Outer radius of circumstellar disc (AU)
    R_in: float                # Inner radius of circumbinary disc (AU)
    R_out: float               # Outer radius of circumbinary disc (AU)
    init_theta_cs_c: float     # Initial angle between L_cs and L_c (radians)
    init_theta_cs_cb: float    # Initial angle between L_cs and L_cb (radians)
    init_theta_c_cb: float     # Initial angle between L_c and L_cb (radians)


class SecularResonanceSystem:
    """
    Secular precession resonance model from Owen & Lai (2017), MNRAS 469, 2834
    Accreting-companion scenario (Section 4.1) only.

    The circumstellar and circumbinary discs deplete viscously (eq. 37),
    the companion accretes a fraction f of the depleted mass, and the
    truncation radii R_cs, R_in follow algebraically from eqs. (41)-(42).
    Uses solve_ivp for the integration.
    """

    def __init__(self, params: SystemParameters):
        self.params = params

        # Initialize unit vectors (L_hat)
        self.l_cs = np.array([0.0, 0.0, 1.0])
        self.l_c = np.array([0.0,
                             -np.sin(params.init_theta_cs_c),
                              np.cos(params.init_theta_cs_c)])
        self.l_cb = np.array([-np.sin(params.init_theta_cs_cb),
                               0.0,
                               np.cos(params.init_theta_cs_cb)])

        # Normalize
        self.l_cs /= np.linalg.norm(self.l_cs)
        self.l_c  /= np.linalg.norm(self.l_c)
        self.l_cb /= np.linalg.norm(self.l_cb)

        # Precession frequency attributes
        self.Omega_cs_c  = 0.0
        self.Omega_cs_cb = 0.0
        self.Omega_c_cb  = 0.0
        self.Omega_c_cs  = 0.0
        self.Omega_cb_c  = 0.0
        self.Omega_cb_cs = 0.0
        self.Omega_cs    = 0.0
        self.Omega_c     = 0.0
        self.Omega_cb    = 0.0

        # Initial angular momentum magnitudes
        self._update_angular_momenta()

        # Accreting-scenario parameters
        self.t_viscous = 2e5     # years
        self.f_accrete = 0.5     # fraction of depleted mass accreted by companion

        # Initial values needed to evaluate eq. (37) at time t
        self.M_cs_initial = params.M_cs
        self.M_cb_initial = params.M_cb
        self.M_c_initial  = params.M_c

        self.current_time = 0.0

        # History tracking
        self.history = {
            'time': [], 'M_cs': [], 'M_c': [], 'M_cb': [],
            'R_cs': [], 'R_in': [], 'a_c': [],
            'l_cs_x': [], 'l_cs_y': [], 'l_cs_z': [],
            'l_c_x':  [], 'l_c_y':  [], 'l_c_z':  [],
            'l_cb_x': [], 'l_cb_y': [], 'l_cb_z': [],
            'theta_cs_c': [], 'theta_cs_cb': [], 'theta_c_cb': [],
            'Omega_cs': [], 'Omega_c': [], 'Omega_cb': [],
            'Omega_cs_c': [], 'Omega_c_cb': [], 'Omega_cs_cb': [],
            'Omega_r': [], 'theta_Lr_cb': [],
            'L_r_x': [], 'L_r_y': [], 'L_r_z': [],
        }

    # ------------------------------------------------------------------
    # Angular momenta
    # ------------------------------------------------------------------
    def _update_angular_momenta(self):
        """L = (2/3) M sqrt(G M_star R) for Sigma ~ R^{-1} discs."""
        self.L_cs_mag = (2/3) * self.params.M_cs * np.sqrt(G * self.params.M_star * self.params.R_cs)
        self.L_c_mag  =          self.params.M_c  * np.sqrt(G * self.params.M_star * self.params.a_c)
        self.L_cb_mag = (2/3) * self.params.M_cb * np.sqrt(G * self.params.M_star * self.params.R_out)

    # ------------------------------------------------------------------
    # Precession frequencies (eq. 16)
    # ------------------------------------------------------------------
    def _update_precession_frequencies(self, t=None):
        p = self.params
        Omega_K_cs  = np.sqrt(G * p.M_star / p.R_cs**3)
        Omega_K_ac  = np.sqrt(G * p.M_star / p.a_c**3)
        Omega_K_out = np.sqrt(G * p.M_star / p.R_out**3)

        dot_cs_c  = np.dot(self.l_cs, self.l_c)
        dot_cs_cb = np.dot(self.l_cs, self.l_cb)
        dot_c_cb  = np.dot(self.l_c,  self.l_cb)

        self.Omega_cs_c  = (-3/8)  * Omega_K_cs * (p.M_c /p.M_star) * (p.R_cs/p.a_c)**3 * dot_cs_c
        self.Omega_cs_cb = (-3/16) * Omega_K_cs * (p.M_cb/p.M_star) * (p.R_cs**3/(p.R_out*p.R_in**2)) * dot_cs_cb
        self.Omega_c_cb  = (-3/8)  * Omega_K_ac * (p.M_cb/p.M_star) * (p.a_c**3 /(p.R_out*p.R_in**2)) * dot_c_cb
        self.Omega_c_cs  = (-1/4)  * Omega_K_ac * (p.M_cs/p.M_star) * (p.R_cs/p.a_c)**2 * dot_cs_c
        self.Omega_cb_c  = (-3/4)  * Omega_K_out* (p.M_c /p.M_star) * (p.a_c/p.R_in)**2 * dot_c_cb
        self.Omega_cb_cs = (-3/16) * Omega_K_out* (p.M_cs/p.M_star) * (p.R_cs/p.R_in)**2 * dot_cs_cb

        self.Omega_cs = self.Omega_cs_c + self.Omega_cs_cb
        self.Omega_c  = self.Omega_c_cb + self.Omega_c_cs
        self.Omega_cb = self.Omega_cb_c + self.Omega_cb_cs

    # ------------------------------------------------------------------
    # Accreting-scenario parameter update (Section 4.1)
    # ------------------------------------------------------------------
    def _update_masses_and_radii_accreting_time(self, t):
        """
        Viscous depletion (eq. 37), companion accretion, and truncation
        radii (eqs. 41-42) with a_c held fixed.
        """
        p = self.params

        # Eq. (37)
        t_factor = np.sqrt(1 + t / self.t_viscous)
        p.M_cb = self.M_cb_initial / t_factor
        p.M_cs = self.M_cs_initial / t_factor**3

        # Companion accretes fraction f of the depleted mass
        depleted_mass = self.M_cb_initial - p.M_cb
        p.M_c = self.M_c_initial + self.f_accrete * depleted_mass

        # Truncation radii, eqs. (41)-(42); a_c is fixed
        q_ratio = p.M_c / (3 * p.M_star)
        if 0 < q_ratio < 1:
            p.R_cs = p.a_c * (1 - 3 * q_ratio**(1/3))
            p.R_in = p.a_c * (1 + 3 * q_ratio**(1/3))

        # Keep radii physical
        p.R_cs = max(0.1, min(p.R_cs, p.a_c * 0.9))
        p.R_in = max(p.R_cs + 0.5, p.R_in)

        self._update_angular_momenta()

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def _compute_dl_dt(self, t, state):
        """
        dl/dt = Ω_total × l for each of l_cs, l_c, l_cb.
        State: [l_cs(3), l_c(3), l_cb(3)].
        """
        self.l_cs = state[0:3]
        self.l_c  = state[3:6]
        self.l_cb = state[6:9]

        self._update_masses_and_radii_accreting_time(t)
        self._update_precession_frequencies(t)

        dl_cs_dt = np.cross(self.Omega_cs_c * self.l_c + self.Omega_cs_cb * self.l_cb,
                            self.l_cs)
        dl_c_dt  = np.cross(self.Omega_c_cs * self.l_cs + self.Omega_c_cb * self.l_cb,
                            self.l_c)
        dl_cb_dt = np.cross(self.Omega_cb_cs * self.l_cs + self.Omega_cb_c * self.l_c,
                            self.l_cb)

        return np.concatenate([dl_cs_dt, dl_c_dt, dl_cb_dt])

    # ------------------------------------------------------------------
    # Rotating-frame diagnostics (Section 3.1, eq. 24)
    # ------------------------------------------------------------------
    def analyze_rotating_frame_dynamics(self):
        self._update_precession_frequencies()

        Omega_r_Lr = self.Omega_cs_c * self.l_c - self.Omega_c_cb * self.l_cb
        Omega_r = np.linalg.norm(Omega_r_Lr)

        if Omega_r > 0:
            L_r = Omega_r_Lr / Omega_r
            theta_Lr_cb = np.degrees(np.arccos(np.clip(np.dot(L_r, self.l_cb), -1, 1)))
        else:
            L_r = np.zeros(3)
            theta_Lr_cb = 90.0

        return {'Omega_r': Omega_r, 'theta_Lr_cb': theta_Lr_cb, 'L_r': L_r}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def _record_state(self):
        h = self.history
        h['time'].append(self.current_time)
        for key in ('M_cs', 'M_c', 'M_cb', 'R_cs', 'R_in', 'a_c'):
            h[key].append(getattr(self.params, key))

        for prefix, vec in (('l_cs', self.l_cs), ('l_c', self.l_c), ('l_cb', self.l_cb)):
            h[f'{prefix}_x'].append(vec[0])
            h[f'{prefix}_y'].append(vec[1])
            h[f'{prefix}_z'].append(vec[2])

        h['theta_cs_c' ].append(np.degrees(np.arccos(np.clip(np.dot(self.l_cs, self.l_c ), -1, 1))))
        h['theta_cs_cb'].append(np.degrees(np.arccos(np.clip(np.dot(self.l_cs, self.l_cb), -1, 1))))
        h['theta_c_cb' ].append(np.degrees(np.arccos(np.clip(np.dot(self.l_c,  self.l_cb), -1, 1))))

        for key in ('Omega_cs', 'Omega_c', 'Omega_cb',
                    'Omega_cs_c', 'Omega_c_cb', 'Omega_cs_cb'):
            h[key].append(getattr(self, key))

        rot = self.analyze_rotating_frame_dynamics()
        h['Omega_r'].append(rot['Omega_r'])
        h['theta_Lr_cb'].append(rot['theta_Lr_cb'])
        h['L_r_x'].append(rot['L_r'][0])
        h['L_r_y'].append(rot['L_r'][1])
        h['L_r_z'].append(rot['L_r'][2])

    # ------------------------------------------------------------------
    # Integrator
    # ------------------------------------------------------------------
    def evolve(self, total_time: float, f_accrete: float = 0.18,
               t_viscous: float = 2e5, max_step: float = 1000.0,
               rtol: float = 1e-8, atol: float = 1e-10):
        """
        Evolve the accreting-companion system with solve_ivp.

        Parameters
        ----------
        total_time   : float  Total evolution time (yr)
        f_accrete    : float  Fraction of depleted mass accreted by companion
        t_viscous    : float  Viscous timescale for disc depletion (yr)
        max_step     : float  Maximum integrator step (yr)
        rtol, atol   : float  Relative and absolute tolerances
        """
        self.f_accrete = f_accrete
        self.t_viscous = t_viscous

        state0 = np.concatenate([self.l_cs, self.l_c, self.l_cb])

        def ode_func(t, y):
            return self._compute_dl_dt(t, y)

        # Record t=0
        self.current_time = 0.0
        self._update_masses_and_radii_accreting_time(0.0)
        self._update_precession_frequencies(0.0)
        self._record_state()

        print(f"Solving accreting-scenario ODE with solve_ivp ...")
        print(f"  Time range : 0 to {total_time/1e6:.2f} Myr")
        print(f"  Max step   : {max_step:.0f} yr")
        print(f"  Tolerances : rtol={rtol:.0e}, atol={atol:.0e}")
        print(f"  f_accrete  : {f_accrete}")
        print(f"  t_viscous  : {t_viscous:.1e} yr")

        solution = solve_ivp(
            ode_func,
            [0, total_time],
            state0,
            method='DOP853',
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            dense_output=True,
        )

        if not solution.success:
            print(f"Warning: Integration failed: {solution.message}")

        # Post-process
        prev_ratio = None
        resonance_crossed = False

        for i, (t, state) in enumerate(zip(solution.t, solution.y.T)):
            self.current_time = t
            self.l_cs = state[0:3]
            self.l_c  = state[3:6]
            self.l_cb = state[6:9]

            # Renormalize against numerical drift
            self.l_cs /= np.linalg.norm(self.l_cs)
            self.l_c  /= np.linalg.norm(self.l_c)
            self.l_cb /= np.linalg.norm(self.l_cb)

            self._update_masses_and_radii_accreting_time(t)
            self._update_precession_frequencies(t)
            self._record_state()

            if not resonance_crossed and i > 0:
                curr_ratio = self.history['Omega_cs'][-1] / self.history['Omega_c'][-1]
                if prev_ratio is not None and (prev_ratio - 1) * (curr_ratio - 1) < 0:
                    resonance_crossed = True
                    print(f"\n*** RESONANCE CROSSING at t = {t/1e6:.2f} Myr ***")
                    print(f"  Ω_cs/Ω_c  = {curr_ratio:.3f}")
                    print(f"  θ_cs,cb   = {self.history['theta_cs_cb'][-1]:.1f}°")
                    rot = self.analyze_rotating_frame_dynamics()
                    print(f"  θ(L_r,L_cb) = {rot['theta_Lr_cb']:.1f}°")
                prev_ratio = curr_ratio

        print(f"\nIntegration complete. {len(solution.t)} time steps recorded.")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def plot_evolution(self):
        fig, axes = plt.subplots(2, 2, figsize=(9, 5.23))
        time_myr = np.array(self.history['time']) / 1e6

        # Panel: Mass evolution
        ax = axes[1, 1]
        ax.plot(np.log10(time_myr), np.log10(self.history['M_cs']),
                label=r'$M_\mathrm{cs}$ (circumstellar)', linewidth=2)
        ax.plot(np.log10(time_myr), np.log10(self.history['M_c']),
                label=r'$M_\mathrm{c}$ (companion)', linewidth=2)
        ax.plot(np.log10(time_myr), np.log10(self.history['M_cb']),
                label=r'$M_\mathrm{cb}$ (circumbinary)', linewidth=2)
        ax.set_xlim(-3, np.log10(max(time_myr)))
        ax.set_ylim(-7, 0)
        ax.set_xlabel('Time (Myr)', fontsize=16)
        ax.set_ylabel('Mass (M☉)', fontsize=16)
        ax.legend()
        ax.tick_params('both', labelsize=16)

        # Panel: Misalignment angles
        ax = axes[0, 0]
        ax.plot(time_myr, self.history['theta_cs_c'],
                label=r'$\theta_{\mathrm{cs},\mathrm{c}}$', linewidth=2)
        ax.plot(time_myr, self.history['theta_cs_cb'],
                label=r'$\theta_{\mathrm{cs},\mathrm{cb}}$ (inner-outer)',
                linewidth=2, color='red')
        ax.plot(time_myr, self.history['theta_c_cb'],
                label=r'$\theta_{\mathrm{c},\mathrm{cb}}$', linewidth=2)
        ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlim(0, time_myr[-1])
        ax.set_ylim(0, 50)
        ax.tick_params('both', labelsize=16)
        ax.set_xticks([])
        ax.set_ylabel('Misalignment Angle\n(degrees)', fontsize=16)
        ax.legend()

        # Panel: Precession frequencies
        ax = axes[1, 0]
        ax.semilogy(time_myr, np.abs(self.history['Omega_cs']),
                    label=r'$|\Omega_\mathrm{cs}|$', linewidth=2)
        ax.semilogy(time_myr, np.abs(self.history['Omega_c']),
                    label=r'$|\Omega_\mathrm{c}|$', linewidth=2)
        ax.set_xlabel('Time (Myr)', fontsize=16)
        ax.set_ylabel('Precession Frequency\n(rad/yr)', fontsize=16)
        ax.set_xlim(0, max(time_myr))
        ax.legend()
        ax.tick_params('both', labelsize=16)

        # Panel: Radii evolution
        ax = axes[0, 1]
        ax.plot(np.log10(time_myr), self.history['a_c'], linewidth=2,
                color='green', label=r'$a_\mathrm{c}$')
        ax.plot(np.log10(time_myr), self.history['R_cs'], '--',
                label=r'$R_{\mathrm{cs}}$', linewidth=2)
        ax.plot(np.log10(time_myr), self.history['R_in'], '--',
                label=r'$R_{\mathrm{in}}$', linewidth=2)
        ax.set_xlim(-3, np.log10(time_myr[-1]))
        ax.set_xticks([])
        ax.set_ylabel('Radius (AU)', fontsize=16)
        ax.tick_params('both', labelsize=16)
        ax.legend()

        plt.tight_layout()
        plt.show()

    def plot1(self):
        time_myr = np.array(self.history['time']) / 1e6
        plt.figure(figsize=(8.8, 7))
        plt.plot(time_myr, self.history['theta_cs_c'],
                 label=r'$\theta_\mathrm{cs,c}$', linewidth=2)
        plt.plot(time_myr, self.history['theta_cs_cb'],
                 label=r'$\theta_\mathrm{cs,cb}$ (inner-outer)',
                 linewidth=2, color='red')
        plt.plot(time_myr, self.history['theta_c_cb'],
                 label=r'$\theta_\mathrm{c,cb}$', linewidth=2)
        plt.xlabel('Time (Myr)', fontsize=16)
        plt.ylabel('Angle (degrees)', fontsize=16)
        plt.title(r'$t_{\mathrm{viscous}}=2\times10^5$ yr, '
                  r'$f_{\mathrm{accrete}}=0.75$', fontsize=16)
        plt.xlim(0, max(time_myr))
        plt.ylim(0, 70)
        plt.legend(fontsize=14)
        plt.tick_params('both', labelsize=16)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Animation (unchanged from lai10.py)
    # ------------------------------------------------------------------
    def create_simple_animation(self, output_filename='system_evolutionV1098_final.gif',
                                fps=10, duration_seconds=30, max_frames=300,
                                zoom_level=2):
        """
        Animation with the circumbinary disc fixed in the xy-plane.
        Only the circumstellar disc and companion orbit evolve.
        """
        print("\nCreating simple animation (GIF format)...")

        n_points = len(self.history['time'])
        frame_indices = np.linspace(0, n_points - 1,
                                    min(max_frames, n_points), dtype=int)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        max_radius_base = max(self.params.R_out, self.history['a_c'][0] * 2, 100)
        max_radius = max_radius_base / zoom_level

        print(f"  Zoom level : {zoom_level}x")
        print(f"  View radius: {max_radius:.1f} AU")
        print(f"  Circumbinary disc fixed in XY plane (reference)")

        ax.set_xlim([-max_radius * 1.1, max_radius * 1.1])
        ax.set_ylim([-max_radius * 1.1, max_radius * 1.1])
        ax.set_zlim([-max_radius * 1.1, max_radius * 1.1])
        ax.set_xlabel('X (AU)', fontsize=10)
        ax.set_ylabel('Y (AU)', fontsize=10)
        ax.set_zlabel('Z (AU)', fontsize=10)
        ax.view_init(elev=25, azim=45)

        # Circumbinary disc — FIXED in xy-plane
        theta = np.linspace(0, 2 * np.pi, 100)
        cb_ring, = ax.plot(self.params.R_in * np.cos(theta),
                           self.params.R_in * np.sin(theta),
                           np.zeros_like(theta),
                           'r-', linewidth=2, alpha=0.7,
                           label='Circumbinary (fixed)')

        r_grid = np.linspace(0, self.params.R_in, 10)
        theta_grid = np.linspace(0, 2 * np.pi, 20)
        R, Theta = np.meshgrid(r_grid, theta_grid)
        ax.plot_surface(R * np.cos(Theta), R * np.sin(Theta),
                        np.zeros_like(R * np.cos(Theta)),
                        alpha=0.1, color='red', zorder=0)

        # Evolving elements
        cs_ring,    = ax.plot([], [], [], 'b-',  linewidth=2.5, alpha=0.9,
                              label='Circumstellar (evolving)')
        comp_orbit, = ax.plot([], [], [], 'g--', linewidth=1.5, alpha=0.8,
                              label='Companion orbit')
        comp_point, = ax.plot([], [], [], 'go',  markersize=8, label='Companion')
        ax.plot([0], [0], [0], 'y*', markersize=14, label='Star')

        cs_vec, = ax.plot([], [], [], 'b-', linewidth=2.5, alpha=0.7,
                          label=r'$\vec{L}_\mathrm{cs}$')
        c_vec,  = ax.plot([], [], [], 'g-', linewidth=2.5, alpha=0.7,
                          label=r'$\vec{L}_\mathrm{c}$')

        vec_scale = max_radius * 0.4
        ax.plot([0, 0], [0, 0], [0, vec_scale], 'r--', linewidth=1.5,
                alpha=0.4, label=r'$\vec{L}_\mathrm{cb}$ (z-axis)')

        ax.legend(loc='upper right')

        time_text  = ax.text2D(0.02, 0.95, '', transform=ax.transAxes,
                               fontsize=16, fontweight='bold')
        angle_text = ax.text2D(0.02, 0.90, '', transform=ax.transAxes,
                               fontsize=16)
        info_text  = ax.text2D(0.02, 0.85, '', transform=ax.transAxes,
                               fontsize=16, color='gray')

        def get_tilted_ring_points(radius, normal_vector):
            theta = np.linspace(0, 2 * np.pi, 100)
            circle = np.array([radius * np.cos(theta),
                               radius * np.sin(theta),
                               np.zeros_like(theta)])
            normal = normal_vector / (np.linalg.norm(normal_vector) + 1e-10)
            z_axis = np.array([0, 0, 1])

            if np.allclose(np.abs(np.dot(normal, z_axis)), 1.0):
                rot_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]) \
                    if np.dot(normal, z_axis) < 0 else np.eye(3)
            else:
                v = np.cross(z_axis, normal)
                s = np.linalg.norm(v)
                c = np.dot(z_axis, normal)
                vx = np.array([[0, -v[2], v[1]],
                               [v[2], 0, -v[0]],
                               [-v[1], v[0], 0]])
                rot_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s**2))

            rotated = np.dot(rot_matrix, circle)
            return rotated[0, :], rotated[1, :], rotated[2, :]

        def update(frame_idx):
            idx = frame_indices[frame_idx]
            t_myr = self.history['time'][idx] / 1e6

            l_cs = np.array([self.history['l_cs_x'][idx],
                             self.history['l_cs_y'][idx],
                             self.history['l_cs_z'][idx]])
            l_c  = np.array([self.history['l_c_x'][idx],
                             self.history['l_c_y'][idx],
                             self.history['l_c_z'][idx]])
            l_cs /= (np.linalg.norm(l_cs) + 1e-10)
            l_c  /= (np.linalg.norm(l_c)  + 1e-10)

            R_cs = self.history['R_cs'][idx]
            a_c  = self.history['a_c'][idx]

            cs_x, cs_y, cs_z = get_tilted_ring_points(R_cs, l_cs)
            cs_ring.set_data(cs_x, cs_y)
            cs_ring.set_3d_properties(cs_z)

            orb_x, orb_y, orb_z = get_tilted_ring_points(a_c, l_c)
            comp_orbit.set_data(orb_x, orb_y)
            comp_orbit.set_3d_properties(orb_z)

            phase = 2 * np.pi * (t_myr / 10)
            orbit_circle = np.array([a_c * np.cos(phase),
                                     a_c * np.sin(phase),
                                     0])
            z_axis = np.array([0, 0, 1])
            if not np.allclose(np.abs(np.dot(l_c, z_axis)), 1.0):
                v = np.cross(z_axis, l_c)
                s = np.linalg.norm(v)
                c = np.dot(z_axis, l_c)
                vx = np.array([[0, -v[2], v[1]],
                               [v[2], 0, -v[0]],
                               [-v[1], v[0], 0]])
                rot_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s**2))
                comp_pos = np.dot(rot_matrix, orbit_circle)
            else:
                comp_pos = -orbit_circle if np.dot(l_c, z_axis) < 0 else orbit_circle

            comp_point.set_data([comp_pos[0]], [comp_pos[1]])
            comp_point.set_3d_properties([comp_pos[2]])

            cs_vec.set_data([0, vec_scale * l_cs[0]],
                            [0, vec_scale * l_cs[1]])
            cs_vec.set_3d_properties([0, vec_scale * l_cs[2]])

            c_vec.set_data([0, vec_scale * l_c[0]],
                           [0, vec_scale * l_c[1]])
            c_vec.set_3d_properties([0, vec_scale * l_c[2]])

            angle_cs_cb = np.degrees(np.arccos(np.clip(np.dot(l_cs, z_axis), -1, 1)))
            angle_c_cb  = np.degrees(np.arccos(np.clip(np.dot(l_c,  z_axis), -1, 1)))

            time_text.set_text(f'Time: {t_myr:.2f} Myr')
            angle_text.set_text(
                fr'$\theta_\mathrm{{cs,cb}}: {angle_cs_cb:.1f}°  |  '
                fr'\theta_\mathrm{{c,cb}}: {angle_c_cb:.1f}°$')
            info_text.set_text('Disc tilt relative to fixed circumbinary plane')

            return [cs_ring, comp_orbit, comp_point, cs_vec, c_vec,
                    time_text, angle_text, info_text]

        n_frames = len(frame_indices)
        print(f"Creating animation with {n_frames} frames ...")

        anim = FuncAnimation(fig, update, frames=n_frames,
                             interval=1000 / fps, blit=False, repeat=True)

        print(f"Saving animation to {output_filename} ...")
        anim.save(output_filename, writer='pillow', fps=fps, dpi=120)
        print(f"Animation saved successfully!")

        plt.close(fig)


# ----------------------------------------------------------------------
# Preset for HD 142527 (from lai10.py)
# ----------------------------------------------------------------------
def setup_hd142527_system():
    return SystemParameters(
        M_star=1,
        M_c=5e-6,
        M_cs=4.34e-7,
        M_cb=0.04,
        a_c=30,
        R_cs=27,
        R_in=31.0,
        R_out=85,
        init_theta_cs_c=np.radians(5.0),
        init_theta_cs_cb=np.radians(5.0),
        init_theta_c_cb=np.radians(5.0),
    )


def main():
    print("=" * 80)
    print("Secular Precession Resonance Model — Accreting Companion")
    print("Owen & Lai (2017) MNRAS, 469, 2834")
    print("=" * 80)

    params = setup_hd142527_system()
    system = SecularResonanceSystem(params)

    print(f"\nInitial parameters (HD 142527-like):")
    print(f"  M_star = {params.M_star} M☉")
    print(f"  M_c    = {params.M_c} M☉")
    print(f"  M_cs   = {params.M_cs} M☉")
    print(f"  M_cb   = {params.M_cb} M☉")
    print(f"  a_c    = {params.a_c} AU")
    print(f"  Initial misalignment = {np.degrees(params.init_theta_cs_cb):.1f}°")

    print("\nEvolving for 5 Myr ...")
    system.evolve(total_time=5e6, f_accrete=0.48,
                  t_viscous=2e5, max_step=5000.0)

    print("\nFinal state:")
    print(f"  M_c    = {system.params.M_c:.4f} M☉")
    print(f"  M_cs   = {system.params.M_cs:.4f} M☉")
    print(f"  M_cb   = {system.params.M_cb:.4f} M☉")
    print(f"  a_c    = {system.params.a_c:.1f} AU")
    print(f"  R_cs   = {system.history['R_cs'][-1]:.1f} AU")
    print(f"  R_in = {system.history['R_in'][-1]:.1f} AU")
    print(f"  Final misalignment θ_cs,cb = {system.history['theta_cs_cb'][-1]:.1f}°")
    
    system.plot_evolution()
    
    print("\n" + "="*80)
    print("Interpretation of L_r and θ(L_r, L_cb):")
    print("="*80)

    system.create_simple_animation(output_filename='system_evolutionV1098_final.gif', fps=10, duration_seconds=30, max_frames=300, zoom_level=2)

    print("\n" + "="*80)
    print("Animation saved as 'system_evolutionV1098_final.gif'")
    print("""
    How L_r relates to misalignment evolution:
    
    1. Rotating frame definition (Equation 24):
       Ω_r L_r = Ω_cs,c * l_c - Ω_c,cb * l_cb
       
       This defines a coordinate system where the precession of the circumstellar
       disc and companion about each other is removed.
    
    2. When Ω_cs,c ≈ Ω_c,cb (resonance condition):
       - Ω_r becomes small
       - L_r points in a direction determined by l_c and l_cb
       - θ(L_r, L_cb) approaches 90° at resonance
    
    3. Why θ(L_r, L_cb) → 90° is important:
       - The torque on the circumbinary disc from the inner system is maximized
         when L_r is perpendicular to L_cb
       - This allows efficient transfer of angular momentum and growth of
         misalignment between the circumstellar disc and circumbinary disc
       
    4. Connection to the plots:
       - When θ(L_r, L_cb) is near 90°, θ_cs,cb grows rapidly
       - The resonance crossing (Ω_cs/Ω_c = 1) occurs when θ(L_r, L_cb) crosses 90°
       - After resonance, the misalignment can reach 60-70° as seen in the paper
    
    This matches Figure 1 in Owen & Lai (2017), showing the resonance mechanism
    for generating large misalignments in protoplanetary systems.
    """)

if __name__ == "__main__":
    main()