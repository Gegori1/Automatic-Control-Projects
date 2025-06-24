# %%
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# System parameters
M = 1.0   # car mass
m = 0.1   # pendulum mass
l = 0.5   # length of pendulum's center of mass
g = 9.81  # gravity force
d = 0.1   # car's damping coefficient

# Define fuzzy model
# The membership function is given by z1 = sin(theta)/theta
# We assume theta is in [-pi/2, pi/2]
# For small angles: z1 = {~1; theta~=0}
# For larger larger values: z1 = {max(~0.6366); theta~=-pi/2 | pi/2}

# Membership function
a = np.sin(np.pi / 2) / (np.pi / 2)

# Linear Model 1 (theta is close to 0)
A1 = np.array([
    [0, 1, 0, 0],
    [(M + m) * g / (M * l), 0, 0, 0],
    [0, 0, 0, 1],
    [-m * g / M, 0, 0, -d / M]
])

B1 = np.array([
    [0],
    [-1 / (M * l)],
    [0],
    [1 / M]
])

# Linear Model 2 (theta is close to +/- pi/2)
A2 = np.array([
    [0, 1, 0, 0],
    [a * (M + m) * g / (M * l), 0, 0, 0],
    [0, 0, 0, 1],
    [-m * g / M, 0, 0, -d / M]
])

B2 = np.array([
    [0],
    [-a / (M * l)],
    [0],
    [1 / M]
])

n = A1.shape[0]
m_in = B1.shape[1]

print("--- T-S Model Matrices ---")
print("A1:\n", A1)
print("B1:\n", B1)
print("A2:\n", A2)
print("B2:\n", B2)
print("--------------------------\n")


# Define stability variables and Control
P = cp.Variable((n, n), symmetric=True)
M1 = cp.Variable((m_in, n))
M2 = cp.Variable((m_in, n))

# Stability LMIs. Find P and Ms
epsilon = 1e-6 # for stability to ensure strict stability
lmi1 = P >> epsilon * np.eye(n)
lmi2 = A1 @ P + P @ A1.T + B1 @ M1 + (B1 @ M1).T << -epsilon * np.eye(n)
lmi3 = A2 @ P + P @ A2.T + B2 @ M2 + (B2 @ M2).T << -epsilon * np.eye(n)

# The sum condition (LMI 4)
G12 = A1 @ P + B1 @ M2 + A2 @ P + B2 @ M1
lmi4 = G12 + G12.T << -epsilon * np.eye(n)


# Stability Problem
constraints = [lmi1, lmi2, lmi3, lmi4]
prob = cp.Problem(cp.Minimize(0), constraints)

print("Solving LMI problem...")
prob.solve(solver=cp.SCS)

# %% Results
if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
    print("Feasible solution found for the LMIs.\n")
    P_sol = P.value
    M1_sol = M1.value
    M2_sol = M2.value

    # Calculate controller gains K = M * P^-1
    P_inv = np.linalg.inv(P_sol)
    K1 = M1_sol @ P_inv
    K2 = M2_sol @ P_inv

    print("--- Results ---")
    print("P matrix (Lyapunov Matrix):\n", P_sol)
    print("\nK1 (Controller Gain for Rule 1):\n", K1)
    print("\nK2 (Controller Gain for Rule 2):\n", K2)
    print("---------------")

else:
    print("LMIs are not feasible. Stability cannot be guaranteed with this method.")
    K1, K2 = None, None

# %% Closed-Loop System simulation
if K1 is not None and K2 is not None:
    
    # Simulation parameters
    dt = 0.02
    T_final = 100
    time = np.arange(0, T_final, dt)
    
    # Initial condition: pendulum is slightly displaced
    x0 = np.array([0.3, 0.0, 0.0, 0.0]) # 0.2 radians is ~11 degrees
    x = np.copy(x0)
    
    # Store history for plotting
    x_hist = np.zeros((len(time), n))
    u_hist = np.zeros(len(time))

    for i in range(len(time)):
        x_hist[i, :] = x
        
        theta = x[0]
        
        # Check theta to avoid division by zero
        if np.abs(theta) < 1e-9:
            z1 = 1.0
        else:
            z1 = np.sin(theta) / theta
        
        # Calculate membership function values
        h1 = (z1 - a) / (1 - a)
        h2 = 1 - h1
        
        # Ensure they are bounded between 0 and 1
        h1 = np.clip(h1, 0, 1)
        h2 = np.clip(h2, 0, 1)

        # Calculate fuzzy controller output
        u = (h1 * (K1 @ x) + h2 * (K2 @ x))[0]
        u_hist[i] = u
        
        # Nonlinear equations of motion
        theta_dot = x[1]
        pos_dot = x[3]
        pos_ddot = (u + m*l*(theta_dot**2)*np.sin(theta) - m*g*np.sin(theta)*np.cos(theta)) / (M + m*(1 - np.cos(theta)**2))
        theta_ddot = (-u*np.cos(theta) - m*l*(theta_dot**2)*np.sin(theta)*np.cos(theta) + (M+m)*g*np.sin(theta)) / (l*(M + m*(1 - np.cos(theta)**2)))
        
        # Updated x_dot
        x_dot = np.array([
            theta_dot,
            theta_ddot,
            pos_dot,
            pos_ddot
        ])
        
        x = x + dt * x_dot

    # %% Plot Results
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Plot Pendulum Angle
    axs[0].plot(time, x_hist[:, 0], label=r'$\theta$ (angle)')
    axs[0].set_ylabel('Angle (rad)')
    axs[0].legend()
    axs[0].set_title('Inverted Pendulum State Trajectories')
    
    # Plot Cart Position
    axs[1].plot(time, x_hist[:, 2], label=r'$pos$ (position)')
    axs[1].set_ylabel('Position (m)')
    axs[1].legend()

    # Plot Control Input
    axs[2].plot(time, u_hist, label='u (force)', color='r')
    axs[2].set_ylabel('Force (N)')
    axs[2].set_xlabel('Time (s)')
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()

# %%
