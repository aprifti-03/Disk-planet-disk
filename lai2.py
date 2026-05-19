import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict
from scipy.integrate import solve_ivp
import warnings

# Constants
G = 4*np.pi**2  # in units of AU^3/(M_sun * yr^2)

@dataclass
class SystemParameters:
    """Container for system parameters"""
    M_star: float  # Primary star mass (M_sun)
    M_c: float  # Companion mass (M_sun)
    M_cs: float  # Circumstellar disc mass (M_sun)
    M_cb: float  # Circumbinary disc mass (M_sun)
    a_c: float  # Companion orbital separation (AU)
    R_cs: float  # Outer radius of circumstellar disc (AU)
    R_in: float  # Inner radius of circumbinary disc (AU)
    R_out: float  # Outer radius of circumbinary disc (AU)
    init_theta_cs_c: float  # Initial angle between L_cs and L_c (radians)
    init_theta_cs_cb: float  # Initial angle between L_cs and L_cb (radians)
    init_theta_c_cb: float  # Initial angle between L_c and L_cb (radians)

class SecularResonanceSystem:
    """
    Implements the secular precession resonance model from Owen & Lai (2017)
    Now using solve_ivp for better numerical integration
    """
    
    def __init__(self, params: SystemParameters):
        self.params = params
        
        # Initialize unit vectors (L_hat)
        # L_cs along z-axis
        self.l_cs = np.array([0.0, 0.0, 1.0])
        # L_c tilted in y-z plane
        self.l_c = np.array([0.0, -np.sin(params.init_theta_cs_c), np.cos(params.init_theta_cs_c)])
        # L_cb tilted in x-z plane
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
        self.t_viscous = 2e5  # years
        self.f_accrete = 0.5  # Fraction of depleted mass accreted by companion
        self.migration_timescale = None
        self.scenario = 'accreting'
        
        # Store initial parameters for time-dependent evolution
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
        # L = (2/3) M * sqrt(G M_star R) for discs with Sigma ~ R^{-1}
        self.L_cs_mag = (2/3) * self.params.M_cs * np.sqrt(G * self.params.M_star * self.params.R_cs)
        self.L_c_mag = self.params.M_c * np.sqrt(G * self.params.M_star * self.params.a_c)
        self.L_cb_mag = (2/3) * self.params.M_cb * np.sqrt(G * self.params.M_star * self.params.R_out)
    
    def _update_precession_frequencies(self, t=None):
        """Calculate precession frequencies from equation (16)"""
        # Keplerian frequencies
        Omega_K_cs = np.sqrt(G * self.params.M_star / self.params.R_cs**3)
        Omega_K_ac = np.sqrt(G * self.params.M_star / self.params.a_c**3)
        Omega_K_out = np.sqrt(G * self.params.M_star / self.params.R_out**3)
        
        # Dot products
        dot_cs_c = np.dot(self.l_cs, self.l_c)
        dot_cs_cb = np.dot(self.l_cs, self.l_cb)
        dot_c_cb = np.dot(self.l_c, self.l_cb)
        
        # Precession frequencies from equation (16)
        # These are the scalars that multiply the perturbing body's angular momentum vector
        self.Omega_cs_c = (-3/8) * Omega_K_cs * (self.params.M_c / self.params.M_star) * \
                          (self.params.R_cs / self.params.a_c)**3 * dot_cs_c
        
        self.Omega_cs_cb = (-3/16) * Omega_K_cs * (self.params.M_cb / self.params.M_star) * \
                           (self.params.R_cs**3 / (self.params.R_out * self.params.R_in**2)) * dot_cs_cb
        
        self.Omega_c_cb = (-3/8) * Omega_K_ac * (self.params.M_cb / self.params.M_star) * \
                          (self.params.a_c**3 / (self.params.R_out * self.params.R_in**2)) * dot_c_cb
        
        self.Omega_c_cs = (-1/4) * Omega_K_ac * (self.params.M_cs / self.params.M_star) * \
                          (self.params.R_cs / self.params.a_c)**2 * dot_cs_c
        
        # Note: In the paper, Ω_cb,c and Ω_cb,cs are defined differently
        # For circumbinary disc, it precesses about both companion and circumstellar disc
        self.Omega_cb_c = (-3/4) * Omega_K_out * (self.params.M_c / self.params.M_star) * \
                          (self.params.a_c / self.params.R_in)**2 * dot_c_cb
        
        self.Omega_cb_cs = (-3/16) * Omega_K_out * (self.params.M_cs / self.params.M_star) * \
                           (self.params.R_cs / self.params.R_in)**2 * dot_cs_cb
        
        # For diagnostic purposes - total precession rates (not used directly in evolution)
        self.Omega_cs = self.Omega_cs_c + self.Omega_cs_cb
        self.Omega_c = self.Omega_c_cb + self.Omega_c_cs
        self.Omega_cb = self.Omega_cb_c + self.Omega_cb_cs
    
    def _compute_dl_dt(self, t, state):
        """
        Compute the rate of change of unit vectors using equation (7-9)
        dl/dt = Ω_total × l, where Ω_total is the sum of precession vectors
        
        State vector: [l_cs_x, l_cs_y, l_cs_z, l_c_x, l_c_y, l_c_z, l_cb_x, l_cb_y, l_cb_z]
        """
        # Extract current unit vectors
        l_cs = state[0:3]
        l_c = state[3:6]
        l_cb = state[6:9]
        
        # Update current state
        self.l_cs = l_cs
        self.l_c = l_c
        self.l_cb = l_cb
        
        # Update parameters if they evolve with time (for accreting/migrating scenarios)
        self._update_parameters_with_time(t)
        
        # Update precession frequencies
        self._update_precession_frequencies(t)
        
        # Construct precession vectors (Ω_vector)
        # Each body precesses about the sum of the other bodies' angular momentum vectors,
        # weighted by the appropriate precession frequency
        
        # Ω_cs_vector = Ω_cs,c * l_c + Ω_cs,cb * l_cb
        Omega_cs_vector = self.Omega_cs_c * l_c + self.Omega_cs_cb * l_cb
        dl_cs_dt = np.cross(Omega_cs_vector, l_cs)
        
        # Ω_c_vector = Ω_c,cs * l_cs + Ω_c,cb * l_cb
        Omega_c_vector = self.Omega_c_cs * l_cs + self.Omega_c_cb * l_cb
        dl_c_dt = np.cross(Omega_c_vector, l_c)
        
        # Ω_cb_vector = Ω_cb,cs * l_cs + Ω_cb,c * l_c
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
        """Update masses and radii for accreting companion scenario (Section 4.1)"""
        # Disc evolution from equation (37)
        t_factor = np.sqrt(1 + t / self.t_viscous)
        
        M_cb_new = self.M_cb_initial / t_factor
        M_cs_new = self.M_cs_initial / t_factor**3
        
        # Companion accretes fraction f of depleted mass
        depleted_mass = self.M_cb_initial - M_cb_new
        M_c_new = self.M_c_initial + self.f_accrete * depleted_mass
        
        # Update parameters
        self.params.M_cb = M_cb_new
        self.params.M_cs = M_cs_new
        self.params.M_c = M_c_new
        
        # Update truncation radii from equations (41)-(42)
        q_ratio = self.params.M_c / (3 * self.params.M_star)
        if q_ratio > 0 and q_ratio < 1:
            self.params.R_cs = (self.params.a_c * (1 - 3 * q_ratio**(1/3)))
            self.params.R_in = (self.params.a_c * (1 + 3 * q_ratio**(1/3)))
        
        # Ensure radii are physical
        self.params.R_cs = max(0.1, min(self.params.R_cs, self.params.a_c * 0.9))
        self.params.R_in = max(self.params.R_cs + 0.5, self.params.R_in)
        
        self._update_angular_momenta()
    
    def _update_masses_and_radii_migrating_time(self, t):
        """Update masses and radii for migrating companion scenario (Section 4.2)"""
        if self.migration_timescale is None:
            return
        
        # Migration from equation (43)
        a_c_new = self.a_c_initial * np.exp(-t / self.migration_timescale)
        self.params.a_c = max(0.5, a_c_new)
        
        # Update truncation radii
        q_ratio = self.params.M_c / (3 * self.params.M_star)
        if q_ratio > 0 and q_ratio < 1:
            self.params.R_cs = self.params.a_c * (1 - 3 * q_ratio**(1/3))
            self.params.R_in = self.params.a_c * (1 + 3 * q_ratio**(1/3))
        
        # Mass transfer between discs from equations (44)-(45)
        # This is approximate - in reality, mass transfer rate is more complex
        # Here we assume mass is transferred from inner to outer disc
        dM_cs = -self.params.M_cs * (1 - np.exp(-t / self.migration_timescale))
        dM_cb = -dM_cs
        
        self.params.M_cs = self.M_cs_initial + dM_cs
        self.params.M_cb = self.M_cb_initial + dM_cb
        
        # Ensure masses are non-negative
        self.params.M_cs = max(0, self.params.M_cs)
        self.params.M_cb = max(0, self.params.M_cb)
        
        self._update_angular_momenta()
    
    def analyze_rotating_frame_dynamics(self):
        """
        Calculate rotating frame quantities from Section 3.1
        
        The rotating frame is defined such that the precession of the circumstellar
        disc and companion about each other is removed. In this frame, the effective
        precession axis L_r is given by:
        
        L_r = (Ω_cs,c * l_c - Ω_c,cb * l_cb) / |Ω_cs,c * l_c - Ω_c,cb * l_cb|
        
        When Ω_cs,c ≈ Ω_c,cb (resonance), L_r points in a direction that can cause
        significant growth of misalignment with L_cb.
        """
        self._update_precession_frequencies()
        
        # Equation (24): Ω_r L_r = Ω_cs,c * l_c - Ω_c,cb * l_cb
        Omega_r_Lr = self.Omega_cs_c * self.l_c - self.Omega_c_cb * self.l_cb
        Omega_r = np.linalg.norm(Omega_r_Lr)
        
        if Omega_r > 0:
            L_r = Omega_r_Lr / Omega_r
            # Angle between L_r and L_cb - when this is 90°, the torque is maximized
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
        
        # Store unit vectors
        self.history['l_cs_x'].append(self.l_cs[0])
        self.history['l_cs_y'].append(self.l_cs[1])
        self.history['l_cs_z'].append(self.l_cs[2])
        self.history['l_c_x'].append(self.l_c[0])
        self.history['l_c_y'].append(self.l_c[1])
        self.history['l_c_z'].append(self.l_c[2])
        self.history['l_cb_x'].append(self.l_cb[0])
        self.history['l_cb_y'].append(self.l_cb[1])
        self.history['l_cb_z'].append(self.l_cb[2])
        
        # Calculate misalignment angles
        theta_cs_c = np.degrees(np.arccos(np.clip(np.dot(self.l_cs, self.l_c), -1, 1)))
        theta_cs_cb = np.degrees(np.arccos(np.clip(np.dot(self.l_cs, self.l_cb), -1, 1)))
        theta_c_cb = np.degrees(np.arccos(np.clip(np.dot(self.l_c, self.l_cb), -1, 1)))
        
        self.history['theta_cs_c'].append(theta_cs_c)
        self.history['theta_cs_cb'].append(theta_cs_cb)
        self.history['theta_c_cb'].append(theta_c_cb)
        
        # Store precession frequencies (these are now guaranteed to be defined)
        self.history['Omega_cs'].append(self.Omega_cs)
        self.history['Omega_c'].append(self.Omega_c)
        self.history['Omega_cb'].append(self.Omega_cb)
        self.history['Omega_cs_c'].append(self.Omega_cs_c)
        self.history['Omega_c_cb'].append(self.Omega_c_cb)
        self.history['Omega_cs_cb'].append(self.Omega_cs_cb)
        
        # Rotating frame analysis
        rot_frame = self.analyze_rotating_frame_dynamics()
        self.history['Omega_r'].append(rot_frame['Omega_r'])
        self.history['theta_Lr_cb'].append(rot_frame['theta_Lr_cb'])
        self.history['L_r_x'].append(rot_frame['L_r'][0])
        self.history['L_r_y'].append(rot_frame['L_r'][1])
        self.history['L_r_z'].append(rot_frame['L_r'][2])
    
    def evolve(self, total_time: float, scenario: str = 'accreting', 
               migration_timescale: float = None, f_accrete: float = 0.5, 
               t_viscous: float = 2e5, max_step: float = 1000.0,
               rtol: float = 1e-8, atol: float = 1e-10):
        """
        Evolve the system using solve_ivp
        
        Parameters:
        -----------
        total_time : float
            Total evolution time (years)
        scenario : str
            'accreting' or 'migrating'
        migration_timescale : float
            Timescale for migration (years)
        f_accrete : float
            Fraction of depleted mass accreted by companion
        t_viscous : float
            Viscous timescale for disc depletion (years)
        max_step : float
            Maximum time step for integrator (years)
        rtol, atol : float
            Relative and absolute tolerance for integrator
        """
        self.scenario = scenario
        self.f_accrete = f_accrete
        self.t_viscous = t_viscous
        self.migration_timescale = migration_timescale
        
        # Initial state vector
        state0 = np.concatenate([self.l_cs, self.l_c, self.l_cb])
        
        # Define the ODE function with time
        def ode_func(t, y):
            return self._compute_dl_dt(t, y)
        
        # Record initial state (need to compute initial frequencies first)
        self.current_time = 0.0
        self._update_precession_frequencies(0.0)  # Compute initial frequencies
        self._record_state()
        
        # Solve the ODE
        print(f"Solving ODE with solve_ivp...")
        print(f"  Time range: 0 to {total_time/1e6:.1f} Myr")
        print(f"  Max step: {max_step:.0f} years")
        print(f"  Tolerances: rtol={rtol:.0e}, atol={atol:.0e}")
        
        solution = solve_ivp(
            ode_func, 
            [0, total_time], 
            state0,
            method='DOP853',  # High-order adaptive method for non-stiff problems
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            dense_output=True
        )
        
        if not solution.success:
            print(f"Warning: Integration failed: {solution.message}")
        
        # Extract and process results
        times = solution.t
        states = solution.y.T
        
        resonance_crossed = False
        prev_ratio = None
        
        for i, (t, state) in enumerate(zip(times, states)):
            self.current_time = t
            self.l_cs = state[0:3]
            self.l_c = state[3:6]
            self.l_cb = state[6:9]
            
            # Ensure normalization (numerical drift can occur)
            norm_cs = np.linalg.norm(self.l_cs)
            norm_c = np.linalg.norm(self.l_c)
            norm_cb = np.linalg.norm(self.l_cb)
            
            if norm_cs > 0:
                self.l_cs /= norm_cs
            if norm_c > 0:
                self.l_c /= norm_c
            if norm_cb > 0:
                self.l_cb /= norm_cb
            
            # Update parameters at this time
            self._update_parameters_with_time(t)
            
            # Update precession frequencies for recording
            self._update_precession_frequencies(t)
            
            # Record state
            self._record_state()
            
            # Check for resonance crossing
            if not resonance_crossed and i > 0:
                curr_ratio = self.history['Omega_cs'][-1] / self.history['Omega_c'][-1]
                if prev_ratio is not None and (prev_ratio - 1) * (curr_ratio - 1) < 0:
                    resonance_crossed = True
                    print(f"\n*** RESONANCE CROSSING at t = {t/1e6:.2f} Myr ***")
                    print(f"  Ω_cs/Ω_c = {curr_ratio:.3f}")
                    print(f"  θ_cs,cb = {self.history['theta_cs_cb'][-1]:.1f}°")
                    rot = self.analyze_rotating_frame_dynamics()
                    print(f"  θ(L_r, L_cb) = {rot['theta_Lr_cb']:.1f}°")
                prev_ratio = curr_ratio
        
        print(f"\nIntegration complete. {len(times)} time steps recorded.")
    
    def plot_evolution(self):
        """Plot the evolution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        time_myr = np.array(self.history['time']) / 1e6
        
        # Panel 1: Mass evolution
        ax = axes[0, 0]
        ax.plot(np.log10(time_myr), np.log10(self.history['M_cs']), label='M_cs (circumstellar)', linewidth=2)
        ax.plot(np.log10(time_myr), np.log10(self.history['M_c']), label='M_c (companion)', linewidth=2)
        ax.plot(np.log10(time_myr), np.log10(self.history['M_cb']), label='M_cb (circumbinary)', linewidth=2)
        ax.set_xlabel('Time (Myr)')
        ax.set_ylabel('Mass (M☉)')
        ax.set_title('Mass Evolution')
        ax.legend()
        ax.grid(True, alpha=0.5)
        
        # Panel 2: Misalignment angles
        ax = axes[0, 1]
        ax.plot(time_myr, self.history['theta_cs_c'], label='θ_cs,c', linewidth=2)
        ax.plot(time_myr, self.history['theta_cs_cb'], label='θ_cs,cb (inner-outer)', linewidth=2, color='red')
        ax.plot(time_myr, self.history['theta_c_cb'], label='θ_c,cb', linewidth=2)
        ax.set_xlabel('Time (Myr)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Misalignment Angles')
        ax.legend()
        ax.grid(True, alpha=0.5)
        
        # Panel 3: Precession frequencies
        ax = axes[1, 0]
        ax.semilogy(time_myr, np.abs(self.history['Omega_cs']), label='|Ω_cs|', linewidth=2)
        ax.semilogy(time_myr, np.abs(self.history['Omega_c']), label='|Ω_c|', linewidth=2)
        ax.set_xlabel('Time (Myr)')
        ax.set_ylabel('Frequency (rad/yr)')
        ax.set_title('Precession Frequencies')
        ax.legend()
        ax.grid(True, alpha=0.5)
        
        # Panel 4: Radii evolution
        ax = axes[1, 1]
        ax.plot(np.log10(time_myr), self.history['R_cs'], label='R_cs (inner disc outer edge)', linewidth=2)
        ax.plot(np.log10(time_myr), self.history['R_in'], label='R_in (outer disc inner edge)', linewidth=2)
        ax.set_xlabel('Time (Myr)')
        ax.set_ylabel('Radius (AU)')
        ax.set_title('Radii Evolution')
        ax.legend()
        ax.grid(True, alpha=0.5)

        # Add note about interpretation 
        plt.tight_layout()
        plt.show()

def setup_hd142527_system():
    """Setup parameters for HD 142527 from the paper"""
    params = SystemParameters(
        M_star=2.2,      # 1.7 M_sun
        M_c=0.0005,       # 0.124 M_sun companion
        M_cs=0.000005,      # 0.0027 M_sun circumstellar disc
        M_cb=0.25,       # 0.3 M_sun circumbinary disc
        a_c=66.15,      # 30 AU separation
        R_cs=52,       # 12.48 AU
        R_in=80,      # 139.99 AU
        R_out=550,                       # 48 AU
        init_theta_cs_c=np.radians(5.0),
        init_theta_cs_cb=np.radians(5.0),
        init_theta_c_cb=np.radians(5.0)
    )
    return params

def main():
    print("=" * 80)
    print("Secular Precession Resonance Model")
    print("Owen & Lai (2017) MNRAS, 469, 2834")
    print("=" * 80)
    
    # Run accreting companion scenario
    print("\n" + "="*80)
    print("Scenario 1: Accreting Companion with Viscously Depleting Discs")
    print("Using solve_ivp for accurate integration")
    print("="*80)
    
    params = setup_hd142527_system()
    system = SecularResonanceSystem(params)
    
    print(f"Initial parameters (HD 142527-like):")
    print(f"  M_star = {params.M_star} M☉")
    print(f"  M_c = {params.M_c} M☉")
    print(f"  M_cs = {params.M_cs} M☉")
    print(f"  M_cb = {params.M_cb} M☉")
    print(f"  a_c = {params.a_c} AU")
    print(f"  Initial misalignment: {np.degrees(params.init_theta_cs_cb):.1f}°")
    
    print("\nEvolving for 4.5 Myr...")
    system.evolve(total_time=3e6, scenario='accreting',
                  f_accrete= 0.015, t_viscous=2e5, max_step=5000.0)
    
    print("\nFinal state:")
    print(f"  M_c = {system.params.M_c:.4f} M☉")
    print(f"  M_cs = {system.params.M_cs:.4f} M☉")
    print(f"  M_cb = {system.params.M_cb:.4f} M☉")
    print(f"  a_c = {system.params.a_c:.1f} AU")
    print(f"  Final misalignment θ_cs,cb = {system.history['theta_cs_cb'][-1]:.1f}°")
    
    system.plot_evolution()
    
    print("\n" + "="*80)
    print("Interpretation of L_r and θ(L_r, L_cb):")
    print("="*80)
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