import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation

# Parameters
N = 100   
beta = 1.0   #contact rate
gamma = 0.5  #recovery rate 
delta_t = 0.01
T = 30
steps = int(T / delta_t)
initial_state = (90, 10)
num_paths = 5


# Function: Simulate one stochastic SIR path
def simulate_stochastic_sir():
    S, I = [initial_state[0]], [initial_state[1]]
    for t in range(steps):
        s, i = S[-1], I[-1]
        probs = np.array([
            beta * s * i / N * delta_t,      # Infection: (s-1, i+1)
            gamma * i * delta_t,             # Recovery: (s, i-1)
            1 - (beta * s * i / N + gamma * i) * delta_t  # No change
        ])
        probs = np.clip(probs, 0, 1)
        probs = probs / np.sum(probs)

        transition = np.random.choice(['infection', 'recovery', 'none'], p=probs)
        if transition == 'infection' and s > 0:
            s -= 1
            i += 1
        elif transition == 'recovery' and i > 0:
            i -= 1
        S.append(s)
        I.append(i)
    return S, I

# Function: Deterministic solution using Euler method
def deterministic_sir():
    S, I, R = [initial_state[0]], [initial_state[1]], [0]
    for _ in range(steps):
        s, i, r = S[-1], I[-1], R[-1]
        ds = -beta * s * i / N
        di = beta * s * i / N - gamma * i
        dr = gamma * i
        S.append(s + ds * delta_t)
        I.append(i + di * delta_t)
        R.append(r + dr * delta_t)
    return S, I, R

# Plot sample paths
time = np.linspace(0, T, steps + 1)
plt.figure(figsize=(12, 6))
print(beta/gamma)

# Stochastic paths
for _ in range(num_paths):
    S_stoch, I_stoch = simulate_stochastic_sir()
    plt.plot(time, I_stoch, alpha=0.5, label='Stochastic I(t)')

# Deterministic path
S_det, I_det, R_det = deterministic_sir()
plt.plot(time, I_det, color='black', linewidth=2, label='Deterministic I(t)')

plt.title('Stochastic vs Deterministic SIR Model (I(t))')
plt.xlabel('Time')
plt.ylabel('Number of Infected Individuals')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()