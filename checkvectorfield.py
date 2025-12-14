import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.colors as cola
from mpl_toolkits.axes_grid1 import make_axes_locatable

enviroment= {"width":6, "length":50, "obstacles":np.array([[10,3],[20,3],[30,3],[40,3]]), "obstacles_radie":0.25*np.ones(4), "borders_y": [0,6], "exit": np.array([[0,50],[6,50]]), "entery": np.array([[0,50],[6,50]]), "flow": np.array([0.5,0.5]), "intensity":0.5, "max_agents":200}

# Create a 2D grid
x = np.arange(13, 17, 0.2)
y = np.arange(1, 5, 0.2)

X, Y = np.meshgrid(x, y)

def get_b_grad_b(pos_a, pos_b, v_b, dt, eps = 1e-10):
    """
    Args:
        pos_a (dim,)
        pos_b (n, dim)
        v_b (n, dim)
        dt (float)
    Returns:
        b (n,)
        grad_b (n, dim)
    """
    r_ab = pos_a - pos_b
    y_b = v_b * dt
    d_1 = np.sum(r_ab ** 2, axis=1)
    d_2 = np.sum((r_ab - y_b)**2, axis=1)
    d_3 = np.sum(y_b ** 2, axis=1)
    b = 1/2 * np.sqrt(np.maximum(d_1 + d_2 - d_3, eps**2))
    grad_b = (2 * r_ab - y_b) / (4 * b[:, np.newaxis])
    return b, grad_b

def get_acc_border(pos_a, y_border, person_radie=0.155, A=5, B=0.2):

    d_y = pos_a[1] - y_border

    force = np.zeros(2)
    force[1] = (A / B) * np.exp(- np.clip(np.abs(d_y)-person_radie,0,None) / B) * np.sign(d_y)

    return force


def get_force_obstacles(pos_a, pos_obs, e0, radie_obs=0.3, person_radie=0.155, A=10, B=0.2, eps=1e-10):
    """
    Docstring for get_force_obstacles
    
    :param pos_a: (dim) Description
    :param pos_obs: (obs, dim) Description
    :param e0: (dim) Description
    :param radie_obs: (1 or obs) Description
    :param A: (1) Description
    :param B: (1) Description
    :param eps: (1) Description
    :returns: (dim) Description
    """
    
    if pos_obs.size == 0:
        return np.zeros(pos_a.shape)
    if pos_obs.shape[0]<3:
        obs_indices=np.arange(pos_obs.shape[0],dtype=int)
    else:
        nu = 2
        obs_indices = np.argpartition(np.linalg.norm(pos_obs-pos_a[np.newaxis,:],axis=1), nu)[:nu]

    r = pos_a[np.newaxis, :] - pos_obs[obs_indices]
    r_abs = np.linalg.norm(r, axis=1)
    n = r /(np.maximum(r_abs, eps)[:, np.newaxis])

    forces = (A / B) * np.exp(- np.clip(r_abs-0.25-person_radie,0,None) / B)[:, np.newaxis] * n
    weights = seight(e0, forces,lam=0.35)
    weighted_forces = forces * weights[:, np.newaxis]
    
    total_force = np.sum(weighted_forces, axis=0)
    
    return total_force

def seight_old(e0, f, phi=200/360*3.14, c=0.31):
    if f.size == 0:
        return np.zeros((0,),dtype=float)
    v1 = np.sum(e0[np.newaxis, :] * f, axis=1)
    v2 = np.linalg.norm(f, axis=1)
    cos_phi = np.cos(phi)
    v2 = cos_phi * v2

    weight = np.where(v1 >= v2, 1.0, c)
    return weight

def seight(e0, f, lam=0.31):
    if f.size == 0:
        return np.zeros((0,),dtype=float)
    
    f = f/np.clip(np.linalg.norm(f, axis=1, keepdims=True),1e-10,None)
    
    l = (lam + (1-lam)/2 * (1-(e0[0]*f[:,0]+e0[1]*f[:,1])))

    return  l

def get_acc_people(pos_a, pos, vs, e0, delta_t=0.1, person_radie=0.155, max_distance=3.0, A=1.7, B=0.28, A_2=0.0, B_2=0.05):
    """
    Calculate the social force between all agents with a  distance below max_distance.
    pos_a: (2,) single agent
    pos: (N, 2) positions of all agents
    vs: (N, 2) velocities of all agents
    e0: (2,) desired direction of agent a
    A=1.7, B=0.28
    A=1.7, B=0.28
    """ 
    neighbor_indices = np.zeros(1,dtype=int)
    if neighbor_indices.size == 0:
        return np.zeros(pos_a.shape)
    pos_b = pos[neighbor_indices]
    v_b = vs[neighbor_indices]

    b, grad_b = get_b_grad_b(pos_a, pos_b, v_b, delta_t)
    forces = ((A/B) * np.exp(-np.clip(b-2*person_radie,0,None) / B)[:, np.newaxis] * seight(e0, grad_b)[:, np.newaxis] + (A_2/B_2) * np.exp(-(b-2*person_radie) / B_2)[:, np.newaxis]) * grad_b
    
    total_force = np.sum(forces, axis=0)

    return total_force

"""
# Define vector field: example a simple rotational field F = (-y, x)
def get_acc_border(pos_a, y_border, A=10, B=0.2):

    r_y = pos_a[1] - y_border

    force = np.zeros(2)
    force[1] = (A / B) * np.exp(- np.abs(r_y) / B) * np.sign(r_y)
    return force

def get_b_grad_b(pos_a, pos_b, v_b, dt, eps = 1e-10):
    ""
    Args:
        pos_a (dim,)
        pos_b (n, dim)
        v_b (n, dim)
        dt (float)
    Returns:
        b (n,)
        grad_b (n, dim)
    ""
    r_ab = pos_a - pos_b
    y_b = v_b * dt
    d_1 = np.sum(r_ab ** 2, axis=1)
    d_2 = np.sum((r_ab - y_b)**2, axis=1)
    d_3 = np.sum(y_b ** 2, axis=1)

    b = 1/2 * np.sqrt(d_1 + d_2 - d_3)
    grad_b = (2 * r_ab - y_b) / (4 * b[:, np.newaxis] + eps)

    return b, grad_b

def get_acc_people(pos_a, pos, delta_t=0.2, max_distance=2.0, A=2.1, B=0.3):
    ""
    Calculate the social force between all agents with a  distance below max_distance.
    "" 
    vs = np.zeros(pos.shape)
    neighbor_indices = np.array([1])
    if neighbor_indices.size == 0:
        return np.zeros(pos_a.shape)
    pos_b = pos[neighbor_indices]
    v_b = vs[neighbor_indices]

    b, grad_b = get_b_grad_b(pos_a, pos_b, v_b, delta_t)

    forces = (A / B) * np.exp(-b / B)[:, np.newaxis] * grad_b

    weighted_forces = forces

    total_force = np.sum(weighted_forces, axis=0)

    return total_force

def get_force_obstacles(pos_a, radie_obs=0.3, A=10, B=0.2):
    r = pos_a[np.newaxis, :] - enviroment["obstacles"]
    r_abs = np.linalg.norm(r, axis=1)
    n = r / r_abs[:, np.newaxis]

    forces = (A / B) * np.exp(- (r_abs-radie_obs) / B)[:, np.newaxis] * n

    weighted_forces = forces
    
    total_force = np.sum(weighted_forces, axis=0)
    
    return total_force
"""

def acc_target( relaxation_time=0.7):
    """
    Calculate the desired speed for agents based on its current positions
    and target destinations.
    """
    return (1. / relaxation_time) * (np.array([1,0]) * np.abs(1.7) - np.array([1.3,0]))

posO=np.array(enviroment["obstacles"])
e0=np.array([1,0])
U = np.zeros(X.shape)
V = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pos = np.array([X[i,j], Y[i,j]])
        force = get_force_obstacles(pos,posO,e0)
        poss= np.array([[15.2,3]])
        force += get_acc_border(pos, enviroment["borders_y"][0])
        force += get_acc_people(pos, poss, np.array([[-1.3,0]]),e0)
        force += get_acc_border(pos, enviroment["borders_y"][1])
        force += acc_target()
        U[i,j] = force[0]
        V[i,j] = force[1]


# Magnitude used for coloring and (optionally) normalizing arrow lengths
M = np.sqrt(U**2 + V**2)

fig, ax = plt.subplots(figsize=(5,4))
# Normalize arrow lengths by magnitude (so arrows are visually uniform) by dividing U,V by M,
# but avoid dividing by zero.
eps = 1e-10
U_norm = U / (M + eps)
V_norm = V / (M + eps)

arrow = pat.FancyArrow(
    15.2,
    3.2,
    -1.1,
    0,
    width=0.01,
    head_width=0.3,
    head_length=0.2,
    color="Black",
    alpha=0.6,
    label="velocity"
)
q = ax.quiver(X, Y, U_norm, V_norm, np.clip(M,0,9.82*0.33), angles='xy', cmap='viridis', scale=30, width=0.005)#, norm=cola.LogNorm())

#ax.add_patch(arrow)
ax.add_patch(pat.Circle((15.2,3),0.31/2,alpha=0.6,color="Red",label="Agent"))
#ax.add_patch(pat.Circle((10,3),0.25,alpha=0.6,color="Green",label="Obstacles"))
#ax.add_patch(pat.Circle((20,3),0.25,alpha=0.6,color="Green"))


# Quiver: use M for color; scale controls arrow length (tweak scale for your figure)
ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="3%", pad=0.1)

ax.set_title('Forces with a agent in opposite direction')
plt.colorbar(q, cax=cax, label='Acceleration [$\mathrm{m\,s^{-2}}$]')
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
