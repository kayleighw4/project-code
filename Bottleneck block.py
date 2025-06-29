import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation


num_nodes = 25
num_RBC = 100
np.random.seed(0)
random.seed(0)
np.set_printoptions(precision=3)


def create_edges_for_square_grid(num_nodes):
    n = int(np.sqrt(num_nodes))
    edges = []
    for row in range(n):
        for col in range(n):
            index = row * n + col
            if col < n-1:
                RHS_node = index + 1  # connects every node to its RHS neighbour (if one exists)
                edges.append((index, RHS_node))
            if row < n-1:
                above_node = index + n  # connects every node to the neighbour above (if it exists)
                edges.append((index, above_node))
    return edges, n


def adj_matrix(nodes, edges, dtype=float):
    adj_matrix = np.zeros((nodes, nodes))
    for (i, j) in edges:
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
    return adj_matrix


def generate_RBC_location(num_RBC, edges, L_val):
    # Returns a list of tuples (edge, distance in [0,L]) for each RBC
    RBC_edges = [random.choice(edges) for _ in range(num_RBC)]
    RBC_location = [(RBC_edges[i], L_val*np.random.rand()) for i in range(num_RBC)]  # need to multiply distance by L
    return RBC_location


def calculate_pressure(num_nodes, q0_val, R_e_mat, adj_matrix):
    # Calculate pressure at each node. Returns a vector of length nodes.
    from IPython import embed

    # vector of mass balance at each node i
    q_node_vec = np.zeros((num_nodes, 1))
    q_node_vec[0] = q0_val  # input flow rate
    q_node_vec[num_nodes-1] = -q0_val  # output flow rate

    # Constructing the Q matrix in (2.12) from the paper
    Q_mat = np.zeros_like(adj_matrix)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] == 1:
                Q_mat[i][i] += 1/R_e_mat[i][j]
                Q_mat[i][j] += -1/R_e_mat[i][j]

    # Setting up the linear system of equations
    # Since we know the 4th element of the vector p is zero, we should exclude it from the calculation, leaving 3 linear equations
    Q_mat_red = Q_mat[:num_nodes-1, :num_nodes-1]  # taking the upper left 3x3 matrix from M which is non-singular
    q_node_red = q_node_vec[:num_nodes-1]
    p_red = np.linalg.solve(Q_mat_red, q_node_red)  # the reduced pressure vector. Missing its 4th entry which we already set to 0.

    # Constructing the pressure vector from the solved linear system of equations
    p_vec = np.append(p_red, 0)

    return p_vec


def calculate_flow_rate(p_vec, R_e_mat, adj_matrix):
    p_diff_mat = np.zeros_like(adj_matrix)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] == 1:
                p_diff_mat[i][j] = p_vec[i] - p_vec[j]
    q_mat = p_diff_mat/R_e_mat # flow rate matrix
    return q_mat


edges, n = create_edges_for_square_grid(num_nodes)
adj_matrix = adj_matrix(num_nodes, edges)


# Parameters (Constants in this model)

rho_val = 1050                              # Density of blood in kg/m^3
D_val = 5e-6                                # Diameter of capillary in m
L_val = 50e-6                               # Default length of capillary in m
mu_val = 1.5e-3                             # viscosity of blood in Pa.s
q0_val = (rho_val*np.pi*D_val**2)/(4e3)     # input flow rate in m^3/s

L_mat = np.copy(adj_matrix)                 # length of capillary
L_mat[adj_matrix == 1] = L_val

Lhat_mat = np.copy(adj_matrix)              # length of RBC
Lhat_mat[adj_matrix == 1] = 8e-6

D_mat = np.copy(adj_matrix)                 # diameter of capillary
D_mat[adj_matrix == 1] = D_val

beta_mat = np.copy(adj_matrix)              # apparent intrinsic viscosity
beta_mat[adj_matrix == 1] = 2.7

theta_mat = np.copy(adj_matrix)             # correction coefficient

r_mat = (128*mu_val)/(np.pi*D_mat**4*rho_val)   # specific resistance

R_mat = r_mat*L_mat                         # nominal resistance

V_val = (np.pi*L_val*D_val**2)/4            # volume of capillary

tau = (2*n*(n-1)*V_val*rho_val)/q0_val              # network turnover time - ratio of total network volume to volumetric flow rate at inflow

# Generate RBC locations, place them on random edges, then calculate the number
# of RBCs on each edge; store that in a matrix N_mat[i,j] where i,j are the nodes
RBC_location = generate_RBC_location(num_RBC, edges, L_val)  # (edge, distance) for each RBC
RBC_edges = [RBC_location[i][0] for i in range(num_RBC)]  # just the list of edges without distances along
N_mat = np.zeros((num_nodes, num_nodes))
for i, j in RBC_edges:
    N_mat[i, j] += 1
    N_mat[j, i] += 1

# Graphing
fig, ax = plt.subplots(figsize=(6, 6))
x_coords = []
y_coords = []
for row in range(n):
    for col in range(n):
        x_coords.append(col)
        y_coords.append(row)

node_coords = [(x_coords[i], y_coords[i]) for i in range(num_nodes)]
node_colors = ['b' for _ in range(num_nodes)]
node_scatter = ax.scatter([coord[0] for coord in node_coords],
                          [coord[1] for coord in node_coords],
                          color=node_colors, s=100, zorder=2)

# Initialise text annotations
text_annotations = [ax.text(x, y, '', fontsize=9, ha='center') for x, y in node_coords]

# Draw edges
edge_lines= []
for i, j in edges:
    x1, y1 = node_coords[i]
    x2, y2 = node_coords[j]
    line, = ax.plot([x1, x2], [y1, y2], color='black', zorder=1)
    edge_lines.append(line)
    
# Draw directional arrows
arrow_lines = []

# Scatter plot for RBCs
rbc_scatter = ax.scatter([], [], color='red', s=50, zorder=3, alpha=0.5)
time_annotation = ax.text(0.5, 1.05*n, '', ha='center', fontsize=12)

p_vec = calculate_pressure(num_nodes, q0_val, R_mat, adj_matrix)
q_mat = calculate_flow_rate(p_vec, R_mat, adj_matrix)
G_mat_init = q_mat * r_mat

G_values_init = [G_mat_init[i][j] for i, j in edges]
G_norm = plt.Normalize(vmin=np.min(G_values_init), vmax=np.max(G_values_init)) # np.max(G_values_init)
sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=G_norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Local Pressure Gradient (G)')


def update(frame):
    updated_positions = []
    rbc_colors = []
    edge_colors = []
    cumulative_q = np.zeros(len(edges))
    from IPython import embed

    # (1) Calculate RBC velocity

    H_mat = (N_mat * Lhat_mat) / L_mat
    H_mat[adj_matrix == 0] = 0  # Ensure non-existent edges remain 0
    R_e_mat = R_mat * (1 + H_mat * beta_mat)
    
    blocked_edges = [(0, 1), (5, 6), (15, 16), (20, 21)]
    for i, j in blocked_edges:
        R_e_mat[i][j] = 1e20
        R_e_mat[j][i] = 1e20

    # Update pressure and pressure difference
    p_vec = calculate_pressure(num_nodes, q0_val, R_e_mat, adj_matrix)

    q_mat = calculate_flow_rate(p_vec, R_e_mat, adj_matrix)
    for idx, (i, j) in enumerate(edges):
        cumulative_q[idx] += abs(q_mat[i][j])


    G_mat = q_mat*r_mat  # local pressure gradient (2.15)

    u_hat_mat = (4*q_mat)/(np.pi*rho_val*D_mat**2)  # Recalculate RBC velocity
    
    # (2) Update RBC positions, including bifurcation rule / reset to bottom-left

    for idx, ((start_node, end_node), pos) in enumerate(RBC_location):
        # Update position along the edge
        pos += u_hat_mat[start_node, end_node] * 0.005  # Update based on velocity and time step

        do_bifurcate = False
        if pos > L_mat[start_node, end_node]:  # RBC reaches the next node
            current_node = end_node
            do_bifurcate = True
        if pos < 0:  # RBC reaches the previous node
            current_node = start_node
            do_bifurcate = True

        if do_bifurcate:
            # Update edge RBC count (N)
            N_mat[start_node, end_node] -= 1
            N_mat[end_node, start_node] -= 1

            if current_node == n * n - 1:  # Special case: top-right corner
                current_node = 0  # Reset to the bottom-left corner

            # Find connected edges to the current node
            connected_edges = [(current_node, j) for j in range(num_nodes) if adj_matrix[current_node][j] == 1]

            # Calculate pressure gradients for all connected edges
            pressure_gradients = [(edge, G_mat[edge[0]][edge[1]]) for edge in connected_edges]

            # Select the edge with the largest pressure gradient
            max_gradient = max(pressure_gradients, key=lambda x: x[1])[1]
            max_edges = [edge for edge, gradient in pressure_gradients if gradient == max_gradient]

            # Randomly choose one if there are ties
            next_edge = random.choice(max_edges)
            
            #debug
            if current_node == 6:
                print(f"\nFrame {frame} — RBC at node 6")
                for j in range(num_nodes):
                    if adj_matrix[6][j] == 1:
                        print(f"Edge (6, {j}): u = {u_hat_mat[6][j]:.3e}, G = {G_mat[6][j]:.3e}, p6 = {p_vec[6]:.3e}, pj = {p_vec[j]:.3e}")
                #print(next_edge)


            # Update RBC to the new edge
            start_node, end_node = next_edge

            # Set initial position based on flow direction
            if u_hat_mat[start_node, end_node] > 0:
                pos = 0.0
            else:
                pos = L_mat[start_node, end_node]


            # Update RBC count on the new edge
            N_mat[next_edge[0], next_edge[1]] += 1
            N_mat[next_edge[1], next_edge[0]] += 1

        # Update RBC location

        RBC_location[idx] = ((start_node, end_node), pos)

        # Calculate (x, y) position for plotting
        x1, y1 = node_coords[start_node]
        x2, y2 = node_coords[end_node]
        x = x1 + (pos / L_mat[start_node, end_node]) * (x2 - x1)
        y = y1 + (pos / L_mat[start_node, end_node]) * (y2 - y1)
        updated_positions.append((x, y))

        if u_hat_mat[start_node, end_node] < 0:
            rbc_colors.append('k')
        else:
            rbc_colors.append('r')

    # Normalize local pressure gradient values for colormap
    for i, j in edges:
        G_val = G_mat[i][j]
        if G_val > 0:
            color_val = G_val
        elif G_val < 0:
            color_val = -G_val
        else:
            color_val = 0

        edge_color = cm.jet(G_norm(color_val))
        edge_colors.append(edge_color)


    # Update edge colors
    for line, color in zip(edge_lines, edge_colors):
        line.set_color(color)
        line.set_linewidth(4) 


    # Update scatter plot
    rbc_scatter.set_offsets(updated_positions)
    
    time_annotation.set_text(f't = {frame * 0.005:.3f} s (t/τ = {frame * 0.005 / tau:.2f})')

    # Remove previous arrows
    for arrow in arrow_lines:
        arrow.remove()
    arrow_lines.clear()

    # Add arrows based on G direction and magnitude
    for i, j in edges:
        G_val = G_mat[i][j]
        if G_val == 0:
            continue

        # Determine direction based on sign
        if G_val > 0:
            x_start, y_start = node_coords[i]
            x_end, y_end = node_coords[j]
        else:
            x_start, y_start = node_coords[j]
            x_end, y_end = node_coords[i]
            G_val = -G_val

        # Arrow vector
        dx = x_end - x_start
        dy = y_end - y_start

        arrow = ax.arrow(
            x_start + 0.1*dx, y_start + 0.1*dy,
            dx * 0.6, dy * 0.6,
            head_width=0.15, head_length=0.2,
            fc=cm.jet(G_norm(G_val)), ec=cm.jet(G_norm(G_val)),
            alpha=0.8, linewidth=0.5
        )
        arrow_lines.append(arrow)

    if frame == 0:
        plt.pause(3)  # Pause for 3 seconds
        
    # Return updated plots
    return [rbc_scatter, node_scatter, time_annotation] + edge_lines + text_annotations + arrow_lines

# Animation
ani = FuncAnimation(fig, update, frames=100000, interval=5, blit=False)
plt.show()