# %% 
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# %% System parameters
M_b = 1.0  # car/body's mass
M_w = 1.0  # wheel's mass
b_s = 1.0  # dumping coefficient
k_s = 1.0  # car's spring system coefficient
k_t = 1.0  # tire spring coefficient
r = 0      # road height

# %% Model
A = np.array([
    [0, 1, 0, 0],
    [-k_s/M_b, -b_s/M_b, k_s/M_b, b_s/M_b],
    [0, 0, 0, 1],
    [k_s/M_w, b_s/M_w, (k_s-k_t)/M_w, -b_s/M_w] 
])

B = np.array([
    [0],
    [1e3/M_b],
    [0],
    [-1e3/M_w]
])

n = A.shape[0]
m_in = B.shape[1]

print("--- Model Matrices ---")
print("A:\n", A)
print("B:\n", B)
print("--------------------------\n")

# %% Controller Design
P = cp.Variable((n, n), symmetric=True)
M = cp.Variable((m_in, n))

# LMIs
epsilon = 1e-6
lmi1 = P >> epsilon * np.eye(n)
lmi2 = A @ P + P @ A.T + B @ M + (B @ M).T << -epsilon * np.eye(n)

# Stability Problem
constraints = [lmi1, lmi2]
prob = cp.Problem(cp.Minimize(0), constraints)

# Solution
print("Solving LMI problem...")
prob.solve(solver=cp.MOSEK)


print(prob.status)
# %% Results
if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
    print("Feasible solution found for the LMIs.\n")
    P_sol = P.value
    M_sol = M.value
    
    # Calculate K = M * P^-1
    P_inv = np.linalg.inv(P_sol)
    K = M_sol @ P_inv
    
    print("--- Results ---")
    print("P matrix (Lyapunov Matrix):\n", P_sol)
    print("\nK (Controller Gain for Rule 1):\n", K)
    print("---------------")
    
else:
    print("LMIs are not feasible. Stability cannot be guaranteed with this method.")
    K = None # Set gain to None if no solution is found
    
# %% Closed-Loop System simulation

if K is not None:
    print("\nStarting simulation...")
    
    # Simulation parameters
    dt = 0.02
    T_final = 10
    time = np.arange(0, T_final, dt)
    
    # Initial condition: there wheel hits an edge-bump
    x0 = np.array([0.0, 0.0, 0.0, 0.1])
    x = np.copy(x0)
    
    x_hist = np.zeros((len(time), n))
    u_hist = np.zeros(len(time))
    
    for i in range(len(time)):
        x_hist[i, :] = x
        
        # Controler for the car body actuator
        u = (K @ x)[0]
        u_hist[i] = u
        
        # Controler for the car
        pos_b = x[0]
        pos_b_dot = x[1]
        pos_w = x[2]
        pos_w_dot = x[3]
        
        # Equation of motion
        pos_b_ddot = -1/M_b * (k_s * (pos_b - pos_w) + b_s * (pos_b_dot - pos_w_dot) - 1e3 * u)
        pos_w_ddot = 1/M_w * (k_s * (pos_b - pos_w) + b_s * (pos_b_dot - pos_w_dot) - k_t * (pos_w - r) - 1e3 * u)
        

        x_dot = np.array([
            pos_b_dot,
            pos_b_ddot,
            pos_w_dot,
            pos_w_ddot
        ])
        
        x = x + dt * x_dot
        
# %%
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Body Position
axs[0].plot(time, x_hist[:, 0])
axs[0].set_ylabel('Body position')
axs[0].set_title('')

# Plot Wheel Position
axs[1].plot(time, x_hist[:, 2])
axs[1].set_ylabel('Wheel position')
axs[0].set_title('')

plt.tight_layout()
plt.show()

print("Simulation Finished.")

# %%
