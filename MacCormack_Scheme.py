import numpy as np 

import matplotlib.pyplot as plt 

from matplotlib.animation import FuncAnimation 

import time 

 

class ShockTubeSimulation: 

    def __init__(self, L_driver=3.0, L_driven=9.0, p_driven=101000, pr_ratio=5.0, T_init=300.0, 

                 R=287.0, gamma=1.4, nx=400, cfl=0.8): 

        """ 

        Initialize shock tube simulation parameters 

         

        Parameters: 

        ----------- 

        L_driver : float 

            Length of driver section in meters 

        L_driven : float 

            Length of driven section in meters 

        p_driven : float 

            Pressure in driven section in Pa 

        pr_ratio : float 

            Diaphragm pressure ratio (p_driver/p_driven) 

        T_init : float 

            Initial temperature in K 

        R : float 

            Gas constant in J/kg-K 

        gamma : float 

            Specific heat ratio 

        nx : int 

            Number of grid points 

        cfl : float 

            CFL number for time step calculation 

        """ 

        # Physical constants 

        self.R = R 

        self.gamma = gamma 

         

        # Domain dimensions 

        self.L_driver = L_driver 

        self.L_driven = L_driven 

        self.L_total = L_driver + L_driven 

         

        # Initial conditions 

        self.p_driven = p_driven 

        self.p_driver = pr_ratio * p_driven 

        self.T_init = T_init 

        self.rho_driven = p_driven / (R * T_init) 

        self.rho_driver = self.p_driver / (R * T_init) 

         

        # Grid setup 

        self.nx = nx 

        self.dx = self.L_total / (nx - 1) 

        self.x = np.linspace(0, self.L_total, nx) 

         

        # Diaphragm location 

        self.diaphragm_idx = int(L_driver / self.L_total * (nx - 1)) 

         

        # CFL number 

        self.cfl = cfl 

         

        # Initialize conservative variables 

        self.U = np.zeros((3, nx)) 

        self.F = np.zeros((3, nx)) 

         

        # Time tracking 

        self.time = 0.0 

        self.dt = 0.0 

         

        # Initialize the flow field 

        self.initialize_flow() 

         

        # Calculate theoretical wave speeds 

        self.calculate_theoretical_speeds() 

         

        # Store history for analysis 

        self.history = {'time': [], 'rho': [], 'u': [], 'p': [], 'T': []} 

         

    def initialize_flow(self): 

        """Set up initial conditions in the shock tube""" 

        # Create arrays for primitive variables 

        rho = np.zeros(self.nx) 

        u = np.zeros(self.nx) 

        p = np.zeros(self.nx) 

         

        # Set initial conditions for driver and driven sections 

        for i in range(self.nx): 

            if i <= self.diaphragm_idx: 

                # Driver section 

                rho[i] = self.rho_driver 

                p[i] = self.p_driver 

            else: 

                # Driven section 

                rho[i] = self.rho_driven 

                p[i] = self.p_driven 

         

        # Convert primitive to conservative variables 

        e = p / ((self.gamma - 1) * rho) + 0.5 * u**2 

         

        # Set conservative variables 

        self.U[0] = rho 

        self.U[1] = rho * u 

        self.U[2] = rho * e 

     

    def calculate_theoretical_speeds(self): 

        """Calculate theoretical wave speeds for validation""" 

        # Sound speeds in driver and driven sections 

        a1 = np.sqrt(self.gamma * self.p_driven / self.rho_driven) 

        a4 = np.sqrt(self.gamma * self.p_driver / self.rho_driver) 

         

        # Pressure ratio 

        p41 = self.p_driver / self.p_driven 

         

        # Solve for pressure ratio across shock 

        # Using shock relation equation (iterative method) 

        p21 = 1.0  # Initial guess 

        for _ in range(100): 

            p21_new = (1 + ((self.gamma+1)/2) * (p41-1) /  

                      (1 + ((self.gamma+1)/2) * (p21-1) *  

                       np.sqrt(1 + ((self.gamma+1)/(self.gamma-1)) * (p21-1)/p21))) 

             

            if abs(p21 - p21_new) < 1e-6: 

                break 

            p21 = p21_new 

         

        # Calculate shock speed 

        Ms = np.sqrt(1 + (self.gamma+1)/(2*self.gamma) * (p21-1)) 

        self.shock_speed = Ms * a1 

         

        # Calculate expansion wave speed 

        self.expansion_head_speed = -a4 

        self.expansion_tail_speed = -(2*a4)/(self.gamma+1) * (1 - (p21/p41)**((self.gamma-1)/(2*self.gamma))) 

         

        # Print theoretical results 

        print(f"Theoretical shock speed: {self.shock_speed:.2f} m/s") 

        print(f"Theoretical expansion head speed: {self.expansion_head_speed:.2f} m/s") 

        print(f"Theoretical expansion tail speed: {self.expansion_tail_speed:.2f} m/s") 

     

    def compute_flux(self, U): 

        """Compute flux vector F from conservative variables U""" 

        F = np.zeros_like(U) 

        rho = U[0] 

        u = U[1] / (rho + 1e-16)  # Add small constant to avoid division by zero 

        e = U[2] / (rho + 1e-16) 

         

        # Compute pressure 

        p = (self.gamma - 1) * rho * (e - 0.5 * u**2) 

         

        # Compute fluxes 

        F[0] = rho * u 

        F[1] = rho * u**2 + p 

        F[2] = (rho * e + p) * u 

         

        return F 

     

    def apply_boundary_conditions(self, U): 

        """Apply boundary conditions: zero velocity at walls and zero gradient for other variables""" 

        # Zero velocity at walls 

        U[1, 0] = 0.0 

        U[1, -1] = 0.0 

         

        # Zero gradient for density and energy 

        U[0, 0] = U[0, 1] 

        U[0, -1] = U[0, -2] 

         

        U[2, 0] = U[2, 1] 

        U[2, -1] = U[2, -2] 

         

        return U 

     

    def calculate_time_step(self): 

        """Calculate time step based on CFL condition""" 

        # Extract primitive variables 

        rho = self.U[0] 

        u = self.U[1] / (rho + 1e-16) 

        p = (self.gamma - 1) * (self.U[2] - 0.5 * rho * u**2) 

         

        # Calculate sound speed 

        a = np.sqrt(self.gamma * p / (rho + 1e-16)) 

         

        # Calculate maximum wave speed 

        max_speed = np.max(np.abs(u) + a) 

         

        # Apply CFL condition 

        dt = self.cfl * self.dx / max_speed 

         

        return dt 

     

    def maccormack_step(self): 

        """Perform one step using MacCormack method""" 

        # Calculate time step 

        self.dt = self.calculate_time_step() 

         

        # Predictor step (forward difference) 

        U_pred = np.copy(self.U) 

        F = self.compute_flux(self.U) 

         

        for i in range(1, self.nx-1): 

            U_pred[:, i] = self.U[:, i] - self.dt / self.dx * (F[:, i+1] - F[:, i]) 

         

        # Apply boundary conditions to predicted values 

        U_pred = self.apply_boundary_conditions(U_pred) 

         

        # Corrector step (backward difference) 

        F_pred = self.compute_flux(U_pred) 

         

        for i in range(1, self.nx-1): 

            self.U[:, i] = 0.5 * (self.U[:, i] + U_pred[:, i] -  

                                 self.dt / self.dx * (F_pred[:, i] - F_pred[:, i-1])) 

         

        # Apply boundary conditions to corrected values 

        self.U = self.apply_boundary_conditions(self.U) 

         

        # Update time 

        self.time += self.dt 

     

    def get_primitive_variables(self): 

        """Convert conservative variables to primitive variables""" 

        rho = self.U[0] 

        u = self.U[1] / (rho + 1e-16) 

        e = self.U[2] / (rho + 1e-16) 

        p = (self.gamma - 1) * rho * (e - 0.5 * u**2) 

        T = p / (self.R * rho + 1e-16) 

         

        return rho, u, p, T 

     

    def run_simulation(self, end_time, save_interval=0.0005): 

        """Run simulation up to specified end time""" 

        next_save_time = 0.0 

        start_time = time.time() 

         

        while self.time < end_time: 

            # Run one step 

            self.maccormack_step() 

             

            # Save results at specified intervals 

            if self.time >= next_save_time: 

                rho, u, p, T = self.get_primitive_variables() 

                 

                self.history['time'].append(self.time) 

                self.history['rho'].append(rho.copy()) 

                self.history['u'].append(u.copy()) 

                self.history['p'].append(p.copy()) 

                self.history['T'].append(T.copy()) 

                 

                next_save_time += save_interval 

                 

                print(f"Simulation time: {self.time:.5f} s, dt: {self.dt:.8f} s") 

         

        elapsed = time.time() - start_time 

        print(f"Simulation completed in {elapsed:.2f} seconds") 

        print(f"Number of time steps: {len(self.history['time'])}") 

     

    def plot_results(self, times_to_plot=None): 

        """Plot results at specified times""" 

        if times_to_plot is None: 

            times_to_plot = [self.history['time'][-1]] 

         

        vars_to_plot = { 

            'p': ('Pressure (Pa)', lambda p: p/1000),  # Convert to kPa 

            'rho': ('Density (kg/m³)', lambda rho: rho), 

            'u': ('Velocity (m/s)', lambda u: u), 

            'T': ('Temperature (K)', lambda T: T) 

        } 

         

        # For each time to plot 

        for plot_time in times_to_plot: 

            # Find closest saved time 

            idx = np.argmin(np.abs(np.array(self.history['time']) - plot_time)) 

            actual_time = self.history['time'][idx] 

             

            fig, axs = plt.subplots(2, 2, figsize=(12, 10)) 

            fig.suptitle(f'Shock Tube Simulation Results at t = {actual_time:.4f} s', fontsize=16) 

             

            for i, (var_name, (label, converter)) in enumerate(vars_to_plot.items()): 

                row, col = i // 2, i % 2 

                ax = axs[row, col] 

                 

                var_data = converter(self.history[var_name][idx]) 

                 

                ax.plot(self.x, var_data) 

                ax.set_xlabel('Position (m)') 

                ax.set_ylabel(label) 

                ax.grid(True) 

                 

                # Add diaphragm position 

                ax.axvline(x=self.L_driver, color='r', linestyle='--', alpha=0.5) 

                 

                # For velocity, add horizontal line at zero 

                if var_name == 'u': 

                    ax.axhline(y=0, color='k', linestyle=':', alpha=0.5) 

             

            plt.tight_layout() 

            plt.subplots_adjust(top=0.92) 

            plt.savefig(f'shock_tube_t_{actual_time:.4f}.png', dpi=300) 

            plt.show() 

     

    def create_animation(self, fps=15, filename='shock_tube_animation.mp4'): 

        """Create animation of the simulation results""" 

        fig, axs = plt.subplots(2, 2, figsize=(12, 10)) 

        fig.suptitle('Shock Tube Simulation', fontsize=16) 

         

        vars_to_plot = { 

            'p': ('Pressure (kPa)', lambda p: p/1000), 

            'rho': ('Density (kg/m³)', lambda rho: rho), 

            'u': ('Velocity (m/s)', lambda u: u), 

            'T': ('Temperature (K)', lambda T: T) 

        } 

         

        lines = [] 

        for i, (var_name, (label, _)) in enumerate(vars_to_plot.items()): 

            row, col = i // 2, i % 2 

            ax = axs[row, col] 

            line, = ax.plot([], []) 

            lines.append(line) 

             

            ax.set_xlabel('Position (m)') 

            ax.set_ylabel(label) 

            ax.grid(True) 

             

            # Add diaphragm position 

            ax.axvline(x=self.L_driver, color='r', linestyle='--', alpha=0.5) 

             

            # Set fixed axis limits based on all data 

            min_val = min([min(converter(data)) for data in self.history[var_name]]) 

            max_val = max([max(converter(data)) for data in self.history[var_name]]) 

             

            margin = (max_val - min_val) * 0.1 

            ax.set_xlim(0, self.L_total) 

            ax.set_ylim(min_val - margin, max_val + margin) 

         

        time_text = fig.text(0.5, 0.95, '', ha='center') 

         

        def init(): 

            for line in lines: 

                line.set_data([], []) 

            time_text.set_text('') 

            return lines + [time_text] 

         

        def update(frame): 

            for i, (var_name, (_, converter)) in enumerate(vars_to_plot.items()): 

                data = converter(self.history[var_name][frame]) 

                lines[i].set_data(self.x, data) 

             

            time_text.set_text(f'Time: {self.history["time"][frame]:.4f} s') 

            return lines + [time_text] 

         

        ani = FuncAnimation(fig, update, frames=len(self.history['time']), 

                           init_func=init, blit=True, interval=1000/fps) 

         

        plt.tight_layout() 

        plt.subplots_adjust(top=0.9) 

        ani.save(filename, fps=fps, extra_args=['-vcodec', 'libx264']) 

         

        return ani 

 

def grid_refinement_analysis(end_time=0.01): 

    """Perform grid refinement analysis""" 

    grid_sizes = [100, 200, 400, 800] 

    results = [] 

     

    for nx in grid_sizes: 

        print(f"\nRunning simulation with {nx} grid points") 

        sim = ShockTubeSimulation(nx=nx) 

        sim.run_simulation(end_time) 

         

        # Get final results 

        rho, u, p, T = sim.get_primitive_variables() 

         

        # Store results 

        results.append({ 

            'nx': nx, 

            'dx': sim.dx, 

            'dt': sim.dt, 

            'cfl': sim.cfl, 

            'rho': rho, 

            'u': u, 

            'p': p, 

            'T': T, 

            'x': sim.x, 

            'shock_position': None  # Will calculate below 

        }) 

         

        # Identify shock position at final time 

        dp_dx = np.gradient(p, sim.dx) 

        shock_idx = np.argmax(dp_dx) 

        shock_position = sim.x[shock_idx] 

        results[-1]['shock_position'] = shock_position 

         

        print(f"Shock position at t={end_time}: {shock_position:.4f} m") 

        print(f"Numerical shock speed: {(shock_position - sim.L_driver)/end_time:.2f} m/s") 

        print(f"Theoretical shock speed: {sim.shock_speed:.2f} m/s") 

        print(f"Relative error: {abs(((shock_position - sim.L_driver)/end_time - sim.shock_speed)/sim.shock_speed)*100:.2f}%") 

     

    # Plot comparison of results 

    plt.figure(figsize=(12, 8)) 

     

    vars_to_plot = { 

        'p': ('Pressure (kPa)', lambda p: p/1000), 

        'rho': ('Density (kg/m³)', lambda rho: rho), 

        'u': ('Velocity (m/s)', lambda u: u), 

        'T': ('Temperature (K)', lambda T: T) 

    } 

     

    for i, (var_name, (label, converter)) in enumerate(vars_to_plot.items()): 

        plt.subplot(2, 2, i+1) 

         

        for result in results: 

            plt.plot(result['x'], converter(result[var_name]),  

                    label=f'nx={result["nx"]}') 

         

        plt.xlabel('Position (m)') 

        plt.ylabel(label) 

        plt.title(f'{var_name.capitalize()} at t={end_time}s') 

        plt.axvline(x=3.0, color='r', linestyle='--', alpha=0.5, label='Diaphragm') 

        plt.grid(True) 

         

        if i == 0:  # Only add legend to first plot 

            plt.legend() 

     

    plt.tight_layout() 

    plt.savefig('grid_refinement.png', dpi=300) 

    plt.show() 

     

    # Print convergence table 

    print("\nGrid Refinement Analysis Summary:") 

    print("--------------------------------") 

    print(f"{'Grid Size':>10} | {'dx (m)':>10} | {'dt (s)':>10} | {'Shock Pos. (m)':>15} | {'Error (%)':>10}") 

    print("-" * 65) 

     

    for result in results: 

        shock_speed = (result['shock_position'] - 3.0) / end_time 

        error = abs((shock_speed - results[0]['shock_position']) / results[0]['shock_position']) * 100 

        print(f"{result['nx']:10d} | {result['dx']:10.6f} | {result['dt']:10.8f} | {result['shock_position']:15.6f} | {error:10.2f}") 

 

def main(): 

    # Create simulation object 

    sim = ShockTubeSimulation(nx=400) 

     

    # Run simulation for sufficient time to capture reflections 

    # Calculate time for shock to reach right wall 

    time_to_wall = sim.L_total / sim.shock_speed 

     

    # Run for twice that time to observe reflections 

    end_time = 2.5 * time_to_wall 

    print(f"Estimated time for shock to reach right wall: {time_to_wall:.4f} s") 

    print(f"Running simulation until: {end_time:.4f} s") 

     

    # Run simulation 

    sim.run_simulation(end_time) 

     

    # Plot results at multiple times 

    times_to_plot = [0.005, 0.01, 0.015, end_time] 

    sim.plot_results(times_to_plot) 

     

    # Create animation 

    # sim.create_animation() 

     

    # Perform grid refinement analysis 

    grid_refinement_analysis() 

 

if __name__ == "__main__": 

    main() 
