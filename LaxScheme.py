import numpy as np 

import matplotlib.pyplot as plt 

import time 

 

class ShockTubeSimulation: 

    def __init__(self, L_driver=3.0, L_driven=9.0, p_driven=101000, pr_ratio=5.0, T_init=300.0, 

                 R=287.0, gamma=1.4, nx=400, cfl=0.8): 

        # Physical constants 

        self.R = R 

        self.gamma = gamma 

        # Domain 

        self.L_driver = L_driver 

        self.L_driven = L_driven 

        self.L_total = L_driver + L_driven 

        # Initial conditions 

        self.p_driven = p_driven 

        self.p_driver = pr_ratio * p_driven 

        self.T_init = T_init 

        self.rho_driven = p_driven / (R * T_init) 

        self.rho_driver = self.p_driver / (R * T_init) 

        # Grid 

        self.nx = nx 

        self.dx = self.L_total / (nx - 1) 

        self.x = np.linspace(0, self.L_total, nx) 

        self.diaphragm_idx = int(L_driver / self.L_total * (nx - 1)) 

        # CFL 

        self.cfl = cfl 

        # Conserved variables 

        self.U = np.zeros((3, nx)) 

        # Time 

        self.time = 0.0 

        self.dt = 0.0 

        # Init flow and compute theory 

        self.initialize_flow() 

        self.calculate_theoretical_speeds() 

        # History 

        self.history = {'time': [], 'rho': [], 'u': [], 'p': [], 'T': []} 

 

    def initialize_flow(self): 

        rho = np.zeros(self.nx) 

        u = np.zeros(self.nx) 

        p = np.zeros(self.nx) 

        for i in range(self.nx): 

            if i <= self.diaphragm_idx: 

                rho[i] = self.rho_driver 

                p[i] = self.p_driver 

            else: 

                rho[i] = self.rho_driven 

                p[i] = self.p_driven 

        e = p / ((self.gamma - 1) * rho) + 0.5 * u**2 

        self.U[0] = rho 

        self.U[1] = rho * u 

        self.U[2] = rho * e 

 

    def calculate_theoretical_speeds(self): 

        a1 = np.sqrt(self.gamma * self.p_driven / self.rho_driven) 

        p41 = self.p_driver / self.p_driven 

        p21 = 1.0 

        for _ in range(100): 

            p21_new = (1 + ((self.gamma+1)/2)*(p41-1) / 

                       (1 + ((self.gamma+1)/2)*(p21-1)* 

                        np.sqrt(1 + ((self.gamma+1)/(self.gamma-1))*(p21-1)/p21))) 

            if abs(p21 - p21_new) < 1e-6: 

                break 

            p21 = p21_new 

        Ms = np.sqrt(1 + (self.gamma+1)/(2*self.gamma)*(p21-1)) 

        self.shock_speed = Ms * a1 

        # Expansion wave speeds omitted for brevity 

        print(f"Theoretical shock speed: {self.shock_speed:.2f} m/s (from diaphragm at x={self.L_driver} m)") 

 

    def compute_flux(self, U): 

        rho = U[0] 

        u = U[1] / (rho + 1e-16) 

        e = U[2] / (rho + 1e-16) 

        p = (self.gamma - 1) * rho * (e - 0.5*u**2) 

        F = np.zeros_like(U) 

        F[0] = rho * u 

        F[1] = rho * u**2 + p 

        F[2] = (rho*e + p) * u 

        return F 

 

    def apply_boundary_conditions(self, U): 

        U[1, 0] = 0.0 

        U[1, -1] = 0.0 

        U[0, 0] = U[0, 1]; U[0, -1] = U[0, -2] 

        U[2, 0] = U[2, 1]; U[2, -1] = U[2, -2] 

        return U 

 

    def calculate_time_step(self): 

        rho = self.U[0] 

        u = self.U[1] / (rho + 1e-16) 

        p = (self.gamma - 1)*(self.U[2] - 0.5*rho*u**2) 

        a = np.sqrt(self.gamma * p / (rho + 1e-16)) 

        max_speed = np.max(np.abs(u) + a) 

        return self.cfl * self.dx / max_speed 

 

    def lax_step(self): 

        self.dt = self.calculate_time_step() 

        F = self.compute_flux(self.U) 

        U_new = np.copy(self.U) 

        for i in range(1, self.nx-1): 

            U_new[:, i] = 0.5*(self.U[:, i+1] + self.U[:, i-1]) \ 

                           - 0.5*self.dt/self.dx*(F[:, i+1] - F[:, i-1]) 

        self.U = self.apply_boundary_conditions(U_new) 

        self.time += self.dt 

 

    def get_primitive_variables(self): 

        rho = self.U[0] 

        u = self.U[1] / (rho + 1e-16) 

        e = self.U[2] / (rho + 1e-16) 

        p = (self.gamma - 1)*rho*(e - 0.5*u**2) 

        T = p / (self.R * rho + 1e-16) 

        return rho, u, p, T 

 

    def run_simulation(self, end_time, save_interval=0.0005): 

        next_save = 0.0 

        start = time.time() 

        while self.time < end_time: 

            self.lax_step() 

            if self.time >= next_save: 

                rho, u, p, T = self.get_primitive_variables() 

                self.history['time'].append(self.time) 

                self.history['rho'].append(rho.copy()) 

                self.history['u'].append(u.copy()) 

                self.history['p'].append(p.copy()) 

                self.history['T'].append(T.copy()) 

                next_save += save_interval 

                print(f"t={self.time:.5f}s, dt={self.dt:.5e}s") 

        print(f"Simulation finished: {time.time()-start:.2f}s, steps={len(self.history['time'])}") 

 

    def plot_results(self, times=None): 

        if times is None: 

            times = [self.history['time'][-1]] 

        vars_map = {'p':('Pressure (kPa)', lambda x: x/1000), 

                    'rho':('Density', lambda x: x), 

                    'u':('Velocity', lambda x: x), 

                    'T':('Temperature', lambda x: x)} 

        for t_plot in times: 

            idx = np.argmin(np.abs(np.array(self.history['time'])-t_plot)) 

            fig, axs = plt.subplots(2,2,figsize=(12,10)) 

            fig.suptitle(f"Results at t={self.history['time'][idx]:.4f}s") 

            for i,(var,(lbl,conv)) in enumerate(vars_map.items()): 

                ax = axs[i//2, i%2] 

                data = conv(self.history[var][idx]) 

                ax.plot(self.x, data) 

                ax.axvline(self.L_driver, color='r', linestyle='--') 

                ax.set_xlabel('x (m)'); ax.set_ylabel(lbl) 

                ax.grid(True) 

            plt.tight_layout(); plt.show() 

 

    def grid_refinement_analysis(self, end_time=0.01): 

        grid_sizes = [100, 200, 400, 800] 

        print("\nGrid Refinement Analysis Summary:") 

        print(" Grid Size |    dx    | Shock Pos. | Error (%)") 

        print("----------------------------------------------") 

        for nx in grid_sizes: 

            sim = ShockTubeSimulation(nx=nx) 

            sim.run_simulation(end_time) 

            rho, u, p, T = sim.get_primitive_variables() 

            dp_dx = np.gradient(p, sim.dx) 

            shock_idx = np.argmax(dp_dx) 

            shock_pos = sim.x[shock_idx] 

            # Theoretical shock position 

            theo_pos = sim.L_driver + sim.shock_speed * end_time 

            error = abs((shock_pos - theo_pos)/theo_pos)*100 

            print(f"{nx:9d} | {sim.dx:8.4f} | {shock_pos:9.4f} | {error:9.2f}") 

 

# Main run 

if __name__ == "__main__": 

    sim = ShockTubeSimulation(nx=400) 

    end_time = 0.01 

    sim.run_simulation(end_time) 

    sim.plot_results([end_time]) 

    sim.grid_refinement_analysis(end_time) 
