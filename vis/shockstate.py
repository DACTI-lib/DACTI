import math

# Define the flow_state as a class
class FlowState:
    def __init__(self, rho, u, p):
        self.rho = rho  # Density
        self.u = u      # Velocity
        self.p = p      # Pressure

# Function to compute normal shock
def compute_normal_shock(pre_shock, shock_mach, gamma=1.4, P0=0.0):
    r1 = pre_shock.rho
    p1 = pre_shock.p
    c1 = math.sqrt(gamma * (p1 + P0) / r1)

    r_ratio = 1.0 - 2.0 / (gamma + 1) * (1 - 1 / (shock_mach * shock_mach))
    r2 = r1 / r_ratio
    u2 = 2.0 / (gamma + 1) * (shock_mach - 1 / shock_mach) * c1
    p2 = p1 * (((2 * P0 / p1 + 1) * (1 - r_ratio) * gamma + 1 + r_ratio) / ((r_ratio - 1) * gamma + 1 + r_ratio))

    return FlowState(r2, u2, p2)

# Example usage
gamma = 1.4
P0 = 0.0
pre_shock_state = FlowState(rho=1.0, u=0.0, p=1e5)
shock_mach = 3.0
post_shock_state = compute_normal_shock(pre_shock_state, shock_mach, gamma=gamma, P0=P0)

print(f"Post-shock density: {post_shock_state.rho}")
print(f"Post-shock velocity: {post_shock_state.u}")
print(f"Post-shock pressure: {post_shock_state.p}")
print(f"Post-shock Mach: {post_shock_state.u / math.sqrt(gamma * (post_shock_state.p + P0) / post_shock_state.rho)}")