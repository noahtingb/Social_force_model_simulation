import numpy as np
import matplotlib.pyplot as plt
enviroment= {"width":6, "length":50, "obstacles":np.array([[10,3],[20,3],[30,3],[40,3]]), "obstacles_radie":0.3*np.ones(4), "borders_y": [0,6], "exit": np.array([[0,50],[6,50]]), "entery": np.array([[0,50],[6,50]]), "flow": np.array([0.5,0.5]), "intensity":0.5, "max_agents":200}

# Create a 2D grid
x = np.arange(0, 50, 0.2)
k=np.arange(-1, 1, 0.1)
y = 3-3*k**2*np.sign(k)

X, Y = np.meshgrid(x, y)

# Define vector field: example a simple rotational field F = (-y, x)
def desired_direction(target, pos):
    """
    Calculate the desired direction for agents based on its current positions
    and target destinations.
    Args:
        target (n, dim)
        pos (n, dim)

    Returns:
        e0 (n, dim)
    """
    direction = target - pos
    norm=np.linalg.norm(direction, axis=1, keepdims=True)
    norm=np.maximum(norm, 1e-10)
    return  direction/norm


def getTargets(classification, pos, radie_avoid=0.6, radie_obs=0.6, length=50, wallwidth=6):
    sqrt2 = np.sqrt(2)
    obstacles = enviroment["obstacles"]
    targets = np.zeros(pos.shape)-999.

    for individual in np.zeros(1,dtype=int):
            if classification[individual] == 0:
                    next_obstacles = obstacles[obstacles[:,0] > pos[individual,0]]

                    if next_obstacles.size == 0:
                        target_ind = np.array([length, pos[individual,1]]) 
                    else:

                        closest_next = next_obstacles[np.argmin(next_obstacles[:,0])]
                        if abs(closest_next[1]- pos[individual, 1]) < radie_avoid:
                            if (closest_next[0]- pos[individual, 0]) < radie_avoid:
                                target_ind = pos[individual, :] + np.array([0, radie_avoid if closest_next[1] <= pos[individual, 1] else - radie_avoid])
                            else:
                                target_ind = closest_next + np.array([-radie_avoid, radie_avoid if closest_next[1] <= pos[individual, 1] else -radie_avoid])
                            
                        else:
                            target_ind = np.array([length, pos[individual,1]]) 
            else:
                    next_obstacles = obstacles[obstacles[:,0] < pos[individual,0]]

                    if next_obstacles.size == 0:
                        target_ind = np.array([0, pos[individual,1]]) 
                    else:
                        # Find the one with the smallest x
                        closest_next = next_obstacles[np.argmax(next_obstacles[:,0])]
                        if abs(closest_next[1] - pos[individual, 1]) < radie_avoid:
                            if - (closest_next[0] - pos[individual, 0]) < radie_avoid:
                                target_ind = pos[individual, :] + np.array([0, radie_avoid if closest_next[1] <= pos[individual, 1] else - radie_avoid])
                            else:
                                target_ind = closest_next + np.array([radie_avoid, radie_avoid if closest_next[1] <= pos[individual, 1] else - radie_avoid])
                        
                        else:
                            target_ind = np.array([0, pos[individual,1]]) 
            targets[individual, :] = np.clip(target_ind, a_min=[0,radie_avoid-radie_obs], a_max=[length, wallwidth-(radie_avoid-radie_obs)])
    return targets


U = np.zeros(X.shape)
V = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pos = np.array([X[i,j], Y[i,j]])
        force = desired_direction(getTargets(np.ones(1),np.array([pos])), np.array([pos]))[0]
        U[i,j] = force[0]
        V[i,j] = force[1]



# Magnitude used for coloring and (optionally) normalizing arrow lengths
M = np.sqrt(U**2 + V**2)

fig, ax = plt.subplots(figsize=(6,6))
# Normalize arrow lengths by magnitude (so arrows are visually uniform) by dividing U,V by M,
# but avoid dividing by zero.
eps = 1e-8
U_norm = U / (M + eps)
V_norm = V / (M + eps)

# Quiver: use M for color; scale controls arrow length (tweak scale for your figure)
q = ax.quiver(X, Y, U_norm, V_norm, M, angles='xy', cmap='viridis')
ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Quiver: Rotational field (arrows colored by magnitude)')
plt.colorbar(q, ax=ax, label='|F|')
plt.grid(True)
plt.tight_layout()
plt.show()
