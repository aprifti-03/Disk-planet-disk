"""
Migrating Companion Scenario - Simplified Coupled ODE System
Owen & Lai (2017) MNRAS, 469, 2834 - Section 4.2

Solves the minimal coupled ODE system:
- d(l_cs, l_c, l_cb)/dt from secular precession (equations 7-9)
- da_c/dt = -a_c/τ_migrate (equation 43)
- dM_cs/dt = -M_cs/τ_migrate (equation 44)
- M_cb determined by mass conservation: M_cb = M_cb(0) + M_cs(0) - M_cs(t)
- Radii R_cs and R_in determined algebraically from a_c (equations 41-42)
- Companion mass M_c remains constant
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.integrate import solve_ivp

# Constants
G = 4*np.pi**2  # in units of AU^3/(M_sun * yr^2)

@dataclass
class SystemParameters:
    """Container for system parameters"""
    M_star: float      # Primary star mass (M_sun)
    M_c: float         # Companion mass (M_sun) - FIXED in migrating scenario
    M_cs: float        # Circumstellar disc mass (M_sun)
    M_cb: float        # Circumbinary disc mass (M_sun)
    a_c: float         # Companion orbital separation (AU)
    R_cs: float        # Outer radius of circumstellar disc (AU)
    R_in: float        # Inner radius of circumbinary disc (AU)
    R_out: float       # Outer radius of circumbinary disc (AU)
    init_theta_cs_c: float     # Initial angle between L_cs and L_c (radians)
    init_theta_cs_cb: float    # Initial angle between L_cs and L_cb (radians)
    init_theta_c_cb: float     # Initial angle between L_c and L_cb (radians)

class MigratingResonanceSystem:
    """
    Implements secular precession resonance model for migrating companion
    Solves minimal coupled ODE system for vectors, a_c, and M_cs
    """
    
    def __init__(self, params: SystemParameters):
        self.params = params
        
        # Initialize unit vectors
        self.l_cs = np.array([0.0, 0.0, 1.0])
        self.l_c = np.array([0.0, -np.sin(params.init_theta_cs_c), np.cos(params.init_theta_cs_c)])
        self.l_cb = np.array([-np.sin(params.init_theta_cs_cb), 0.0, np.cos(params.init_theta_cs_cb)])
        
        # Normalize
        self.l_cs /= np.linalg.norm(self.l_cs)
        self.l_c /= np.linalg.norm(self.l_c)
        self.l_cb /= np.linalg.norm(self.l_cb)
        
        # Initialize precession frequencies
        self.Omega_cs_c = 0.0
        self.Omega_cs_cb = 0.0
        self.Omega_c_cb = 0.0
        self.Omega_c_cs = 0.0
        self.Omega_cb_c = 0.0
        self.Omega_cb_cs = 0.0
        self.Omega_cs = 0.0
        self.Omega_c = 0.0
        self.Omega_cb = 0.0
        
        # Store initial parameters
        self.M_cs_initial = params.M_cs
        self.M_cb_initial = params.M_cb
        self.M_c_initial = params.M_c  # Will remain constant
        self.a_c_initial = params.a_c
        
        self.current_time = 0.0
        self.migration_timescale = None
        
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
            'Omega_r': [], 'theta_Lr_cb': []
        }
        
        self._update_angular_momenta()
    
    def _update_angular_momenta(self):
        """Calculate angular momentum magnitudes"""
        self.L_cs_mag = (2/3) * self.params.M_cs * np.sqrt(G * self.params.M_star * self.params.R_cs)
        self.L_c_mag = self.params.M_c * np.sqrt(G * self.params.M_star * self.params.a_c)
        self.L_cb_mag = (2/3) * self.params.M_cb * np.sqrt(G * self.params.M_star * self.params.R_out)
    
    def _update_radii(self):
        """
    Update truncation radii based on current a_c (Equations 41-42)
    These are algebraic relations, not differential equations
    """
        q_ratio = self.params.M_c / (3 * self.params.M_star)
        if 0.01 < q_ratio < 1:
            # Equation (41): Circumstellar disc outer radius
            self.params.R_cs = self.params.a_c * (1 - 3 * q_ratio**(1/3))
            # Equation (42): Circumbinary disc inner radius
            self.params.R_in = self.params.a_c * (1 + 3 * q_ratio**(1/3))
    
        # Ensure radii are physical
        self.params.R_cs = max(0.1, min(self.params.R_cs, self.params.a_c * 0.9))
        self.params.R_in = max(self.params.R_cs + 0.5, min(self.params.R_in, self.params.R_out * 0.9))
    
        # NEW: Set outer radius to be 1.2 times inner radius


    def _update_mass_cb(self):
        """
        Update M_cb from mass conservation (M_cb = M_cb(0) + M_cs(0) - M_cs)
        Since dM_cs/dt = -M_cs/τ, mass is transferred from inner to outer disc
        """
        self.params.M_cb = self.M_cb_initial + (self.M_cs_initial - self.params.M_cs)
    
    def _update_precession_frequencies(self):
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
        
        # Total precession rates (for diagnostics)
        self.Omega_cs = self.Omega_cs_c + self.Omega_cs_cb
        self.Omega_c = self.Omega_c_cb + self.Omega_c_cs
        self.Omega_cb = self.Omega_cb_c + self.Omega_cb_cs
    
    def _compute_derivatives(self, t, state):
        """
        Compute all derivatives for the minimal coupled ODE system
        
        State vector contains 9+1+1 = 11 components:
        state[0:3]  = l_cs (3 components)
        state[3:6]  = l_c  (3 components)
        state[6:9]  = l_cb (3 components)
        state[9]    = a_c  (semi-major axis)
        state[10]   = M_cs (circumstellar disc mass)
        """
        
        # Extract current state
        l_cs = state[0:3]
        l_c = state[3:6]
        l_cb = state[6:9]
        a_c = state[9]
        M_cs = state[10]
        
        # Update current state
        self.l_cs = l_cs
        self.l_c = l_c
        self.l_cb = l_cb
        self.params.a_c = a_c
        self.params.M_cs = M_cs
        
        # Update M_cb from mass conservation (algebraic, not an ODE)
        self._update_mass_cb()
        
        # Update radii algebraically from current a_c (Equations 41-42)
        self._update_radii()
        
        # Update angular momenta
        self._update_angular_momenta()
        
        # Update precession frequencies
        self._update_precession_frequencies()
        
        # 1. Derivatives of unit vectors (dl/dt = Ω × l)
        Omega_cs_vector = self.Omega_cs_c * l_c + self.Omega_cs_cb * l_cb
        dl_cs_dt = np.cross(Omega_cs_vector, l_cs)
        
        Omega_c_vector = self.Omega_c_cs * l_cs + self.Omega_c_cb * l_cb
        dl_c_dt = np.cross(Omega_c_vector, l_c)
        
        Omega_cb_vector = self.Omega_cb_cs * l_cs + self.Omega_cb_c * l_c
        dl_cb_dt = np.cross(Omega_cb_vector, l_cb)
        
        # 2. Derivative of semi-major axis (Equation 43)
        da_c_dt = -a_c / self.migration_timescale
        
        # 3. Derivative of circumstellar disc mass (Equation 44)
        dM_cs_dt = -M_cs / self.migration_timescale
        
        # Combine all derivatives
        derivatives = np.concatenate([
            dl_cs_dt, dl_c_dt, dl_cb_dt,
            [da_c_dt],
            [dM_cs_dt]
        ])
        
        return derivatives
    
    def analyze_rotating_frame_dynamics(self):
        """Calculate rotating frame quantities from Section 3.1"""
        self._update_precession_frequencies()
        
        # Equation (24): Ω_r L_r = Ω_cs,c * l_c - Ω_c,cb * l_cb
        Omega_r_Lr = self.Omega_cs_c * self.l_c - self.Omega_c_cb * self.l_cb
        Omega_r = np.linalg.norm(Omega_r_Lr)
        
        if Omega_r > 0:
            L_r = Omega_r_Lr / Omega_r
            theta_Lr_cb = np.degrees(np.arccos(np.clip(np.dot(L_r, self.l_cb), -1, 1)))
        else:
            theta_Lr_cb = 90.0
        
        return {'Omega_r': Omega_r, 'theta_Lr_cb': theta_Lr_cb}
    
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
        
        # Store precession frequencies
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
    
    def evolve(self, total_time: float, migration_timescale: float = 0.75e6,
               max_step: float = 500.0, rtol: float = 1e-8, atol: float = 1e-10):
        """
        Evolve the minimal coupled system using solve_ivp
        
        Parameters:
        -----------
        total_time : float
            Total evolution time (years)
        migration_timescale : float
            Timescale for companion migration (years)
        max_step : float
            Maximum time step for integrator (years)
        rtol, atol : float
            Relative and absolute tolerance for integrator
        """
        self.migration_timescale = migration_timescale
        
        # Initial state vector: [l_cs, l_c, l_cb, a_c, M_cs] (11 components)
        state0 = np.concatenate([
            self.l_cs, self.l_c, self.l_cb,
            [self.params.a_c],
            [self.params.M_cs]
        ])
        
        # Define the ODE function
        def ode_func(t, y):
            return self._compute_derivatives(t, y)
        
        # Record initial state
        self.current_time = 0.0
        self._update_precession_frequencies()
        self._record_state()
        
        # Solve the coupled ODE system
        print(f"\nSolving MINIMAL COUPLED ODE SYSTEM with solve_ivp...")
        print(f"  State vector dimension: {len(state0)} (9 vectors + a_c + M_cs)")
        print(f"  Migration timescale: {migration_timescale/1e6:.2f} Myr")
        print(f"  Time range: 0 to {total_time/1e6:.1f} Myr")
        print(f"  Companion mass remains constant: {self.params.M_c:.3f} M☉")
        print(f"\nDifferential equations being solved:")
        print(f"  • dl_cs/dt, dl_c/dt, dl_cb/dt (secular precession - Equations 7-9)")
        print(f"  • da_c/dt = -a_c/τ_migrate (Equation 43)")
        print(f"  • dM_cs/dt = -M_cs/τ_migrate (Equation 44)")
        print(f"\nAlgebraic relations (updated at each time step):")
        print(f"  • M_cb = M_cb(0) + M_cs(0) - M_cs(t) (mass conservation)")
        print(f"  • R_cs = a_c·[1 - 3(M_c/3M_star)^(1/3)] (Equation 41)")
        print(f"  • R_in = a_c·[1 + 3(M_c/3M_star)^(1/3)] (Equation 42)")
        
        solution = solve_ivp(
            ode_func, 
            [0, total_time], 
            state0,
            method='DOP853',
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            dense_output=True
        )
        
        if not solution.success:
            print(f"Warning: Integration failed: {solution.message}")
        
        # Process results
        times = solution.t
        states = solution.y.T
        
        resonance_crossed = False
        prev_ratio = None
        
        print("\nProcessing solution...")
        for i, (t, state) in enumerate(zip(times, states)):
            self.current_time = t
            
            # Extract state components
            self.l_cs = state[0:3]
            self.l_c = state[3:6]
            self.l_cb = state[6:9]
            self.params.a_c = state[9]
            self.params.M_cs = state[10]
            
            # Normalize vectors (numerical drift can occur)
            self.l_cs /= np.linalg.norm(self.l_cs)
            self.l_c /= np.linalg.norm(self.l_c)
            self.l_cb /= np.linalg.norm(self.l_cb)
            
            # Update M_cb from mass conservation
            self._update_mass_cb()
            
            # Update radii algebraically from current a_c
            self._update_radii()
            
            # Update angular momenta
            self._update_angular_momenta()
            
            # Update precession frequencies
            self._update_precession_frequencies()
            
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
                    print(f"  a_c = {self.params.a_c:.1f} AU")
                    print(f"  M_cs = {self.params.M_cs:.4f} M☉")
                    print(f"  M_cb = {self.params.M_cb:.4f} M☉")
                    print(f"  R_cs = {self.params.R_cs:.1f} AU")
                    print(f"  R_in = {self.params.R_in:.1f} AU")
                prev_ratio = curr_ratio
        
        print(f"\nIntegration complete. {len(times)} time steps recorded.")
        
        # Check mass conservation
        initial_total = self.M_cs_initial + self.M_cb_initial + self.M_c_initial
        final_total = self.params.M_cs + self.params.M_cb + self.params.M_c
        print(f"\nMass conservation check:")
        print(f"  Initial total mass: {initial_total:.4f} M☉")
        print(f"  Final total mass: {final_total:.4f} M☉")
        print(f"  Relative error: {abs(initial_total - final_total)/initial_total:.2e}")
        print(f"  Mass transferred: {self.M_cs_initial - self.params.M_cs:.4f} M☉")
        
        if resonance_crossed:
            print(f"✓ Resonance crossing occurred during evolution")
        else:
            print("✗ No resonance crossing detected")
    
    def plot_evolution(self):
        """Plot the evolution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        time_myr = np.array(self.history['time']) / 1e6
        
        # Panel 1: Mass evolution
        ax = axes[0, 0]
        ax.plot(np.log10(time_myr), np.log10(self.history['M_cs']), label='M_cs (circumstellar) - ODE', linewidth=2)
        ax.plot(np.log10(time_myr), np.log10(self.history['M_c']), label='M_c (companion) - constant', linewidth=2)
        ax.plot(np.log10(time_myr), np.log10(self.history['M_cb']), label='M_cb (circumbinary) - conservation', linewidth=2)
        ax.set_xlabel('Time (Myr)')
        ax.set_ylabel('Mass (M☉)')
        ax.set_title('Mass Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Misalignment angles
        ax = axes[0, 1]
        ax.plot(time_myr, self.history['theta_cs_c'], label='θ_cs,c', linewidth=2)
        ax.plot(time_myr, self.history['theta_cs_cb'], label='θ_cs,cb (inner-outer)', linewidth=2, color='red')
        ax.plot(time_myr, self.history['theta_c_cb'], label='θ_c,cb', linewidth=2)
        ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Time (Myr)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Misalignment Angles')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Precession frequencies
        ax = axes[1, 0]
        ax.semilogy(time_myr, np.abs(self.history['Omega_cs']), label='|Ω_cs|', linewidth=2)
        ax.semilogy(time_myr, np.abs(self.history['Omega_c']), label='|Ω_c|', linewidth=2)
        ax.semilogy(time_myr, np.abs(self.history['Omega_cb']), label='|Ω_cb|', linewidth=2)
        ax.set_xlabel('Time (Myr)')
        ax.set_ylabel('Frequency (rad/yr)')
        ax.set_title('Precession Frequencies')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        
        # Panel 4: Radii evolution (algebraic relations)
        ax = axes[1, 1]
        ax.plot(time_myr, self.history['a_c'], linewidth=2, color='green', label='a_c (ODE)')
        ax.plot(time_myr, self.history['R_cs'], '--', label='R_cs (Eq. 41)', linewidth=2)
        ax.plot(time_myr, self.history['R_in'], '--', label='R_in (Eq. 42)', linewidth=2)
        ax.set_xlabel('Time (Myr)')
        ax.set_ylabel('Radius (AU)')
        ax.set_title('Radii Evolution - Algebraic Relations')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    

    def print_summary(self):
        """Print a summary of the simulation results"""
        print("\n" + "="*80)
        print("MIGRATING SCENARIO SUMMARY (Minimal ODE System)")
        print("="*80)
        print(f"Total time: {self.current_time/1e6:.2f} Myr")
        print(f"Migration timescale: {self.migration_timescale/1e6:.2f} Myr")
        
        print(f"\nInitial state:")
        print(f"  M_c = {self.M_c_initial:.4f} M☉ (constant)")
        print(f"  M_cs = {self.M_cs_initial:.4f} M☉")
        print(f"  M_cb = {self.M_cb_initial:.4f} M☉")
        print(f"  a_c = {self.a_c_initial:.1f} AU")
        print(f"  R_cs = {self.params.R_cs:.1f} AU")
        print(f"  R_in = {self.params.R_in:.1f} AU")
        print(f"  θ_cs,cb = {self.history['theta_cs_cb'][0]:.1f}°")
        
        print(f"\nFinal state:")
        print(f"  M_c = {self.params.M_c:.4f} M☉ (unchanged)")
        print(f"  M_cs = {self.params.M_cs:.4f} M☉")
        print(f"  M_cb = {self.params.M_cb:.4f} M☉")
        print(f"  a_c = {self.params.a_c:.1f} AU")
        print(f"  R_cs = {self.params.R_cs:.1f} AU")
        print(f"  R_in = {self.params.R_in:.1f} AU")
        print(f"  θ_cs,cb = {self.history['theta_cs_cb'][-1]:.1f}°")
        
        # Mass conservation
        initial_total = self.M_cs_initial + self.M_cb_initial + self.M_c_initial
        final_total = self.params.M_cs + self.params.M_cb + self.params.M_c
        print(f"\nMass conservation: Initial total = {initial_total:.4f} M☉, Final total = {final_total:.4f} M☉")
        print(f"  Mass transferred from inner to outer disc: {self.M_cs_initial - self.params.M_cs:.4f} M☉")
        
        # Find maximum misalignment
        max_misalignment = max(self.history['theta_cs_cb'])
        max_time = self.history['time'][self.history['theta_cs_cb'].index(max_misalignment)] / 1e6
        print(f"\nMaximum misalignment: {max_misalignment:.1f}° at t = {max_time:.2f} Myr")
        
        # Check resonance
        ratio = np.array(self.history['Omega_cs']) / np.array(self.history['Omega_c'])
        if np.min(ratio) <= 1 <= np.max(ratio):
            print("✓ Resonance crossing occurred")
        else:
            print("✗ No resonance crossing detected")

def setup_migrating_system(M_star: float = 2, M_c: float = 0.2,
                          M_cs: float = 0.1, M_cb: float = .05,
                          a_c: float = 200, R_cs: float = 6.90212,
                          R_in: float = 393.0978, R_out: float = 471.7174,
                          init_misalignment: float = 5.0):
    """
    Setup parameters for migrating companion scenario
    
    Following Owen & Lai (2017) Section 4.2:
    - Companion mass remains constant
    - Mass transfer follows dM_cs/dt = -M_cs/τ_migrate
    - Companion orbit decays exponentially
    - Radii determined algebraically from a_c (Equations 41-42)
    """
    params = SystemParameters(
        M_star=M_star,
        M_c=M_c,           # Will remain constant
        M_cs=M_cs,
        M_cb=M_cb,
        a_c=a_c,
        R_cs=R_cs,
        R_in=R_in,
        R_out=R_out,
        init_theta_cs_c=np.radians(init_misalignment),
        init_theta_cs_cb=np.radians(init_misalignment),
        init_theta_c_cb=np.radians(init_misalignment)
    )
    return params

def main():
    print("="*80)
    print("MIGRATING COMPANION SCENARIO - MINIMAL COUPLED ODE SYSTEM")
    print("Owen & Lai (2017) MNRAS, 469, 2834 - Section 4.2")
    print("="*80)
    
    # Get user input for initial misalignment
    init_misalignment = input("\nEnter initial misalignment (degrees) [5.0]: ")
    init_misalignment = float(init_misalignment) if init_misalignment else 5.0
    
    # Setup system with user-defined misalignment
    params = setup_migrating_system(init_misalignment=init_misalignment)
    system = MigratingResonanceSystem(params)
    
    print(f"\nSystem parameters (HD 142527-like):")
    print(f"  M_star = {params.M_star} M☉")
    print(f"  M_c = {params.M_c} M☉ (remains constant)")
    print(f"  M_cs(0) = {params.M_cs} M☉")
    print(f"  M_cb(0) = {params.M_cb} M☉")
    print(f"  a_c(0) = {params.a_c} AU")
    print(f"  Initial misalignment = {init_misalignment:.1f}°")
    
    # Choose migration timescale
    print(f"\nMigration timescale options:")
    print(f"  1. Fast: 0.75 Myr")
    print(f"  2. Moderate: 0.5 Myr (fiducial)")
    print(f"  3. Slow: 1.0 Myr")
    print(f"  4. Custom")
    
    choice = input("Enter choice (1-4): ")
    
    if choice == '1':
        tau_mig = 0.75e6
    elif choice == '2':
        tau_mig = 0.5e6
    elif choice == '3':
        tau_mig = 1e6
    elif choice == '4':
        tau_mig = float(input("Enter migration timescale (Myr): ")) * 1e6
    else:
        tau_mig = 1e6
    
    # Evolve the minimal coupled system
    system.evolve(total_time=4e6, migration_timescale=tau_mig)
    
    # Plot results
    system.plot_evolution()
    
    # Print summary
    system.print_summary()

if __name__ == "__main__":
    main()