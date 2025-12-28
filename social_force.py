import numpy as np

import time
from scipy.constants import Boltzmann as kB 
from tkinter import *    
import json

def dumpa(data, filename="noname.json",make_a_dump=False):
    if make_a_dump:
        with open(filename, "w") as f:
            json.dump(data, f)

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

def get_b_grad_b(pos_a, pos_b, v_b, DT, eps = 1e-10):
    """
    Args:
        pos_a (dim,)
        pos_b (n, dim)
        v_b (n, dim)
        DT (float)
    Returns:
        b (n,)
        grad_b (n, dim)
    """
    r_ab = pos_a - pos_b
    y_b = v_b * DT

    d_1 = np.sqrt(np.sum(r_ab ** 2, axis=1, keepdims=True))
    d_2 = np.sqrt(np.sum((r_ab - y_b)**2, axis=1, keepdims=True))
    d_3 = np.sum(y_b ** 2, axis=1, keepdims=True)

    d_1 = np.maximum(d_1, eps)
    d_2 = np.maximum(d_2, eps)

    b = 1/2 * np.sqrt(np.maximum((d_1 + d_2)**2 - d_3, eps**2))

    grad_b = (d_1 + d_2) / (4 * b) * (r_ab / d_1 + (r_ab - y_b) / d_2)

    return b[:,0], grad_b

def get_acc_border(pos_a, y_border, e0, vel_a, DT_b, person_radie=0.155, A=5, B=0.1):

    d_y = pos_a[1] - y_border

    force = np.zeros(2)

    force[1] = (A / B) * np.exp(- np.clip(np.abs(d_y +  DT_b * vel_a[1])-person_radie, 0, None) / B) * np.sign(d_y)

    if np.abs(force[1])>1e-5:
        force*= seight(e0, force[np.newaxis,:], lam=0.31)

    return force


def get_force_obstacles(pos_a, DT_o, DT, vel_a, pos_obs, e0, radie_obs=0.3, person_radie=0.155, A=5, B=0.2, A1=10, B1=0.1, eps=1e-10):
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
        obs_indices = np.arange(pos_obs.shape[0],dtype=int)
        obs_indices1 = np.arange(pos_obs.shape[0],dtype=int)
    else:
        nu = 2
        obs_indices = np.argpartition(np.linalg.norm(pos_obs-pos_a[np.newaxis,:]-DT_o*vel_a[np.newaxis, :],axis=1), nu)[:nu]
        obs_indices1 = np.argpartition(np.linalg.norm(pos_obs-pos_a[np.newaxis,:]-0*vel_a[np.newaxis, :],axis=1), nu)[:nu]

    r = pos_a[np.newaxis, :]+DT_o*vel_a[np.newaxis, :] - pos_obs[obs_indices]
    r_abs = np.linalg.norm(r, axis=1, keepdims=True)
    n = r /(np.maximum(r_abs, eps))

    r1 = pos_a[np.newaxis, :]+ DT*vel_a[np.newaxis, :] - pos_obs[obs_indices1]
    r1_abs = np.linalg.norm(r1, axis=1, keepdims=True)
    n1 = r1 /(np.maximum(r1_abs, eps))

    forces = (A / B) * np.exp(- np.clip(r_abs-radie_obs[obs_indices]-person_radie, 0, None) / B) * n
    weights = seight(e0, forces, lam=0.31)
    weighted_forces = forces * weights[:, np.newaxis]

    forces1 = (A1 / B1) * np.exp(- np.clip(r1_abs-radie_obs[obs_indices1]-person_radie, 0, None) / B1) * n1
    weights1 = seight(e0, forces1, lam=0.31)
    weighted_forces1 = forces1 * weights1[:, np.newaxis]

    total_force = np.sum(weighted_forces, axis=0) + np.sum(weighted_forces1, axis=0)
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
    
    l = lam + (1-lam)/2 * (1-(e0[0]*f[:,0]+e0[1]*f[:,1]))

    return  l

def get_acc_people(pos_a, pos, vs, e0, DT, delta_T, person_radie=0.155, max_distance=3.0, A=2.1, B=0.28, A_2=7, B_2=0.1):
    """
    Calculate the social force between all agents with a  distance below max_distance.
    pos_a: (2,) single agent
    pos: (N, 2) positions of all agents
    vs: (N, 2) velocities of all agents
    e0: (2,) desired direction of agent a
    A=1.7, B=0.28
    A=1.7, B=0.28
    """ 
    neighbor_indices = get_neighbour_indices(pos, pos_a, max_distance)
    if neighbor_indices.size == 0:
        return np.zeros(pos_a.shape)
    pos_b = pos[neighbor_indices]
    v_b = vs[neighbor_indices]

    b, grad_b = get_b_grad_b(pos_a, pos_b, v_b, DT)

    r = (pos_a[np.newaxis,:]-pos_b)
    b2 = np.linalg.norm(r, axis=1)
    grad_b2 = r / np.maximum(b2[:, np.newaxis], 1e-10)
    #b2, grad_b2 = get_b_grad_b(pos_a, pos_b, v_b, delta_T)
    

    
    forces = (A/B) * np.exp(-np.clip(b-2*person_radie, 0, None) / B)[:, np.newaxis] * seight(e0, grad_b, 0.31)[:, np.newaxis] * grad_b + (A_2/B_2) * np.exp(-(b2-2*person_radie-0.2) / B_2)[:, np.newaxis] * seight(e0, grad_b2, 0.31)[:, np.newaxis]  * grad_b2
    
    total_force = np.sum(forces, axis=0)

    return total_force

def get_neighbour_indices(pos, pos_a, max_radius, eps=1e-8):
    """
    Get the indices of neighboring agents within a certain radius.
    """
    distances = np.linalg.norm(pos - pos_a, axis=1)

    neighbor_indices = np.where((eps < distances) & (distances < max_radius))[0]
    return neighbor_indices

def acc_target(vels, e0, desired_speed=1.34, relaxation_time=0.5):
    """
    Calculate the desired speed for agents based on its current positions
    and target destinations.
    """
    return (1. / relaxation_time) * (e0 * np.abs(desired_speed)[:,np.newaxis] - vels)

def social_force(pos, vs, e0, DT_o, DT_p, DT, DT_b, y_borders, obstacles, obstacles_radie, desired_speeds, person_radie, relaxation_time, max_distance):
    """
    Calculate the total acceleration for all individual agents based on social forces.
    """
    acc = np.zeros(pos.shape)
    # Acceleration towards target
    if pos.shape[0] > 0:
        acc = acc_target(vs, e0, desired_speeds, relaxation_time)

    for index_individual in range(pos.shape[0]):
            
            for y_border in y_borders:
                acc[index_individual, :] += get_acc_border(pos[index_individual,:], y_border, e0[index_individual, :], vs[index_individual, :], DT_b, person_radie)
            
            acc[index_individual, :] += get_force_obstacles(pos[index_individual, :], DT_o, DT, vs[index_individual, :], obstacles, e0[index_individual, :], obstacles_radie, person_radie)
            
            acc[index_individual, :] += get_acc_people(pos[index_individual,:], pos, vs, e0[index_individual, :], DT_p, DT, person_radie, max_distance)

    return acc

def getTargetsNew(classification, pos, length=50):
    targets = np.zeros(pos.shape)-999. 
    for individual in range(pos.shape[0]):
            targets[individual] = np.array([length if classification[individual]==0 else 0, pos[individual,1]])
    return targets

def getTargets(classification, pos, obstacles, radie_extra, radie_person, radie_obs, length, wallwidth=6):
    sqrt2 = np.sqrt(2)
    targets = np.zeros(pos.shape)-999. 
    n_obs = obstacles.shape[0]

    full_list=np.array([-10]+list(obstacles[:,0])+[length+10])

    if n_obs > 0:
        before_feeling = (full_list[1:]-full_list[:-1])/2
    else:
        before_feeling = np.zeros(1)

    if type(radie_obs) != np.ndarray:
        radie_obs = radie_obs * np.ones(n_obs)
    before_feeling = np.clip(before_feeling, radie_person, 10)
    for individual in range(pos.shape[0]):
            if classification[individual] == 0:
                    next_obstacles = obstacles[obstacles[:,0] > pos[individual,0]]
                    n_nobs = next_obstacles.shape[0]
                    target_ind = np.array([length, pos[individual,1]]) 

                    if n_nobs > 0:
                        closest_next = next_obstacles[0,:]
                        radie_avoidx =  radie_person + radie_obs[n_obs-n_nobs]
                        radie_avoidy = radie_avoidx+radie_extra
                        if abs(closest_next[1]- pos[individual, 1]) < radie_avoidy:
                            if (closest_next[0]- pos[individual, 0]) < radie_avoidx:
                                target_ind = pos[individual, :] + np.array([0, radie_avoidy if closest_next[1] <= pos[individual, 1] else - radie_avoidy])
                            elif (closest_next[0]- pos[individual, 0]) < before_feeling[n_obs-n_nobs]:
                                target_ind = closest_next + np.array([-radie_avoidx, radie_avoidy if closest_next[1] <= pos[individual, 1] else -radie_avoidy])
                            
            else:
                    next_obstacles = obstacles[obstacles[:,0] < pos[individual,0]]
                    n_nobs = next_obstacles.shape[0]
                    target_ind = np.array([0, pos[individual,1]]) 

                    if n_nobs > 0:
                        closest_next = next_obstacles[n_nobs-1,:]
                        radie_avoidx =  radie_person + radie_obs[n_nobs-1]
                        radie_avoidy =  radie_avoidx+radie_extra
                        if abs(closest_next[1] - pos[individual, 1]) < radie_avoidy:
                            if - (closest_next[0] - pos[individual, 0]) < radie_avoidx:
                                target_ind = pos[individual, :] + np.array([0, radie_avoidy if closest_next[1] <= pos[individual, 1] else - radie_avoidy])
                            elif - (closest_next[0] - pos[individual, 0]) < before_feeling[n_nobs]:
                                target_ind = closest_next + np.array([radie_avoidx, radie_avoidy if closest_next[1] <= pos[individual, 1] else - radie_avoidy])
                             

            targets[individual] = np.array([length if classification[individual]==0 else 0, pos[individual,1]])
            targets[individual, :] = np.clip(target_ind, a_min=[0, radie_person+radie_extra], a_max=[length, wallwidth-(radie_person+radie_extra)])
    return targets

def sim(delta_t, y_borders, obstacles, obstacles_radie, radie_extra=0.05, radie_person=0.2, partflow = 0.1, people_per_second_per_meter = 0.25, peoples = 10000, length = 50, safe_dist=1, plots_varible={}, fluc_max=0.02, fluc_min=0.02, mean_speed_=1.34, std_speed_=0.28, relaxation_time=0.7, N_skip=100, max_distance=3.0):
    width = y_borders[1] - y_borders[0]
    #fluc max = 0.05, 0.02 old     
    # ----------------------------------
    # for animation
    # ----------------------------------
    maxwindow= 1500
    window_sizex, window_sizey = maxwindow, maxwindow*width//length

    dT_o = 0.5
    dT_p = 0.5
    dT_b = 0.3

    tk = Tk()
    tk.geometry(f'{window_sizex + 20}x{window_sizey + 20}')
    tk.configure(background='#000000')

    canvas = Canvas(tk, background='#ECECEC')  # Generate animation window 
    tk.attributes('-topmost', 0)
    canvas.place(x=10, y=10, height=window_sizey, width=window_sizex)

    def stop_loop(event):
        global running
        running = False
        tk.destroy()

    tk.bind("<Escape>", stop_loop) 
    running = True 

    # ----------------------------------
    # not for animation
    # ----------------------------------

    people_per_second = people_per_second_per_meter * (2*width)
    time_realise = np.linspace(0, peoples*1/people_per_second, peoples)
    time_appred = np.zeros(peoples)-999.

    time_travel = np.zeros(peoples)-999.
    active= np.zeros((peoples), dtype=bool)

    classification_all = np.where(np.random.random(peoples)<partflow, 0, 1)
    
    pos_all = np.zeros((active.size,2))

    p=3
    pos_all[:,1] = (1/(2*p)+(p-1)/p*np.random.random(active.size))*(width)
    pos_all[:,0] = np.where(classification_all==0, -safe_dist, length+safe_dist)

    velocities_all = np.zeros(pos_all.shape)
    velocities_all[:,0] = np.clip(np.random.normal(mean_speed_, std_speed_, velocities_all.shape[0]),0.3,None)*(1-2*classification_all)
    deseried_speed_all = velocities_all[:,0]

    max_iterations = np.ceil((time_realise[-1]+length*2)/delta_t)

    iteration = 0
    
    active[0] = True
    actives = 1
    check_active_person = 1
    
    active_not_in_sim = np.array([],dtype=int)

    last_time = time.time()

    while iteration < max_iterations and (actives > 0 or check_active_person<peoples):
        #relise new persons
        
        if check_active_person<peoples:
            if iteration*delta_t >= time_realise[check_active_person]:
                active[check_active_person]=True
                actives += 1                  
                active_not_in_sim = np.append(active_not_in_sim, check_active_person)
                #print("add:",check_active_person, active_not_in_sim)
                check_active_person += 1     
        index = 0
        while index < active_not_in_sim.size:
            check_index = active_not_in_sim[index]
            if pos_all[check_index,0]>0 and pos_all[check_index,0]<length: # in sim
                time_appred[check_index] = iteration*delta_t
                active_not_in_sim = active_not_in_sim[active_not_in_sim!=check_index]
                #print("delete:",check_index, active_not_in_sim)
            else:
                index+=1
        if actives>300:
            break
        active_indices = np.where(active)[0]
        classification =  classification_all[active_indices]
        velocities = velocities_all[active_indices]
        deseried_speed = deseried_speed_all[active_indices]
        pos = pos_all[active_indices]

        targets = getTargetsNew(classification, pos)
        #targets = getTargets(classification, pos, obstacles, radie_extra, radie_person, obstacles_radie, length)
        e0 = desired_direction(targets, pos)
        acc_social = social_force(pos, velocities, e0, dT_o, dT_p, delta_t, dT_b, y_borders, obstacles, obstacles_radie, deseried_speed, radie_person, relaxation_time, max_distance)

        speed = np.linalg.norm(velocities,  axis=1,keepdims=True)
        fluc = (fluc_max-fluc_min)*np.clip(1-np.abs(speed/(1*deseried_speed[:,np.newaxis]))**4,0,1)+fluc_min
        fluc *= np.sqrt(delta_t)
        fluctation = np.random.normal(0, fluc, size=pos.shape)

        acc = acc_social + fluctation

        acc = np.clip(acc, -9.82, 9.82)
        eps = 1e-12

        a_acc_max = 1.5   # forward acceleration [m/s^2]
        a_bra_max = 9   # braking acceleration [m/s^2]

        acc_n = np.linalg.norm(acc, axis=1, keepdims=True)  # shape (N,1)

        is_braking = np.sum(acc * velocities, axis=1, keepdims=True) < 0  # shape (N,1)

        acc_e = acc / (acc_n + eps)  

        acc = np.where(
            is_braking, 
            acc_e * np.minimum(acc_n, a_bra_max),  # braking
            acc_e * np.minimum(acc_n, a_acc_max)   # forward
        )
        
        velocities += acc * delta_t
        
        speed = np.linalg.norm(velocities,  axis=1,keepdims=True)
        
        for i in range(velocities.shape[0]):
            speed = np.linalg.norm(velocities[i, :])
            if np.abs(speed) > 1.1*np.abs(deseried_speed[i]):
                velocities[i,:] = (velocities[i,:] / speed) *1.1* np.abs(deseried_speed[i])
        
        pos += velocities * delta_t
        pos[:,1] = np.clip(pos[:,1], y_borders[0]+radie_person, y_borders[1]-radie_person)
        # ----------------------------------
        # for animation
        # ----------------------------------
        if running:
            rp = plots_varible.get("rp", 0.2)  # Plotting radius of a particle.
            line_width = plots_varible.get("line_width", 1)  # Width of the arrow line.
            vp = plots_varible.get("vp", 1)  # Length of the arrow indicating the velocity direction.
            rps =  rp / length * window_sizex
            if iteration % N_skip == 0:   
                canvas.delete("all")
                for j in active_indices:
                    canvas.create_oval(
                        (pos_all[j,0] - rp) / length * window_sizex,
                        (pos_all[j,1] - rp) / width * window_sizey,
                        (pos_all[j,0] + rp) / length * window_sizex,
                        (pos_all[j,1] + rp) / width * window_sizey,
                        outline='#FF0000' if classification_all[j]==1 else '#0000FF', 
                        fill='#FF0000' if classification_all[j]==1 else '#0000FF',
                    )
                            
                for j in active_indices:
                    canvas.create_line(
                        pos_all[j,0] / length * window_sizex,
                        pos_all[j,1] / width * window_sizey,
                        (pos_all[j,0] + vp*velocities_all[j,0]) / length * window_sizex,
                        (pos_all[j,1] + vp*velocities_all[j,1]) / width * window_sizey,
                        width=line_width,
                        arrow="last",
                        arrowshape= (rps, 2*rps, rps)
                    )
                

                for j in range(obstacles.shape[0]):
                    canvas.create_oval(
                        (obstacles[j,0] - obstacles_radie[j]) / length * window_sizex, 
                        (obstacles[j,1] - obstacles_radie[j]) / width * window_sizey,
                        (obstacles[j,0] + obstacles_radie[j]) / length * window_sizex, 
                        (obstacles[j,1] + obstacles_radie[j]) / width * window_sizey,
                        outline="#33FF00", 
                        fill="#00FF1E",
                    )

                for j in range(targets.shape[0]):
                    canvas.create_oval(
                        (targets[j,0] - 0.2) / length * window_sizex,
                        (targets[j,1] - 0.2) / width * window_sizey,
                        (targets[j,0] + 0.2) / length * window_sizex,
                        (targets[j,1] + 0.2) / width * window_sizey,
                        outline="#FFF700", 
                        fill="#222222",
                    )
                            
                tk.title(f'time {iteration*delta_t:.1f} s, active agents {actives}, total agents {check_active_person}')
                tk.update_idletasks()
                tk.update()
                now_time=time.time()
                #if now_time-last_time<delta_t:
                #    time.sleep(delta_t-(now_time-last_time))  # Increase to slow down the simulation.    
                #else:
                #    print(f"Warning: the simulation is running slower than real time by {now_time-last_time-delta_t:.2f} seconds.")
                time.sleep(0.001)
                #print(f"Time itr {iteration}: {(now_time-last_time):.5f}s ")
                last_time=now_time
        # ----------------------------------
        # not for animation
        # ----------------------------------

        for i in range(actives):
            if classification[i] == 0:
                if pos[i,0] >= length:
                    active[active_indices[i]] = False
                    time_travel[active_indices[i]] = iteration*delta_t - time_appred[active_indices[i]]
                    actives -= 1
            else:
                if pos[i,0] <= 0:
                    active[active_indices[i]] = False
                    time_travel[active_indices[i]] = iteration*delta_t - time_appred[active_indices[i]]
                    actives -= 1

        classification_all[active_indices] =  classification
        velocities_all[active_indices] = velocities
        deseried_speed_all[active_indices] = deseried_speed
        pos_all[active_indices] = pos

        iteration += 1

        # ----------------------------------
        # for animation
        # ----------------------------------

    if running:
        tk.update_idletasks()
        tk.update()
        tk.destroy() 
    # ----------------------------------
    # not for animation
    # ----------------------------------

    return time_travel, classification_all, deseried_speed_all


#simmoddel obstecales
def run_code_depending_on_part(part="", filename="noname", N_skip_animation=1, dt_=0.1):
    length_, width_ = 50, 6

    rp_ = 0.2  # Plotting radius of a particle.
    vp_ = 0.5  # Length of the arrow indicating the velocity direction.
    line_width = 1  # Width of the arrow line.

    plots_varible = {"rp":rp_, "vp":vp_, "line_width":line_width}

    tries = 3#5
    if part == "number_obs": #part 1
        tries = 5
        obst = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        stds = 0.2*np.ones(obst.size)
        md = 2.
        people_per_second_per_meter_ = 0.25 #0.2
        
    elif part.lower()== "design_choice": #part 2
        tries = 5
        obst = np.array([30, 0, 5])#0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        stds = 0.2*np.ones(obst.size)
        segments = 5
        legthseg = 4
        md = 3.
        people_per_second_per_meter_ = 0.25 #0.2
    #obst = np.array([100, 10, 0])

    elif part.lower() == "std_of_speed": #part 3
        tries = 5
        obst = 20*np.ones(11,dtype=int)#30
        stds = 0.04*np.arange(11)
        segments = 2
        legthseg = 2.5
        md = 2.
        people_per_second_per_meter_ = 0.25 #0.2
    else:
        print("No part selected, running default settings")
        return

    modifywith=True

    avg_times = np.zeros((11, tries))
    avg_std =  np.zeros((11, tries))

    n_people= 1000 #1000
    
    partflow_ = .5
    width_obs = 1

    times_ = np.zeros((obst.size, tries, n_people))-999.
    class_ = np.zeros(times_.shape)-999.
    max_speed_ = np.zeros(times_.shape)-999.

    for obs_index, amount_of_obstacles in enumerate(obst):
        fre_len = 5
        if amount_of_obstacles==0:
            obstacles_array_ = np.zeros((0,2))
            radie_obstacles_ = np.zeros((0,))
        elif amount_of_obstacles==1:
            obstacles_array_ = np.array([[length_/2, width_/2]])
            radie_obstacles_ = 0.25 * np.ones(1)
        elif amount_of_obstacles<20:
            obstacles_array_ = np.array([[(length_)*(i+1)/(amount_of_obstacles+1), width_/2] for i in range(amount_of_obstacles)])
            radie_obstacles_ = 0.25 * np.ones(amount_of_obstacles)
        else:
            obstacles_array_ = np.array([[49, width_/2] for i in range(amount_of_obstacles)])
            radie_obstacles_ = 0.1 * np.ones(amount_of_obstacles)

            if amount_of_obstacles>=500:
                a12=5
                b13=0.5
                b12 = 0.1
                for i in range(15):
                    for j in range(2):
                        obstacles_array_[-1-4*i-j,0] = length_ - b13 * i - a12 
                        obstacles_array_[-1-4*i-j-2,0] =  a12 + i*b13
                        obstacles_array_[-1-4*i-j,1] = width_*(j%2) - (2*j-1)* b12*(2.5-abs(2-i))
                        obstacles_array_[-1-4*i-j-2,1] = width_*(j%2) - (2*j-1)* b12*(2.5-abs(2-i))              
                amount_of_obstacles-=60
            safe_dist = 5
            hole_size = 5
            #segments = 2#5#10

            
            legthseg = 2*0.25*amount_of_obstacles/segments
            if amount_of_obstacles == 20 and segments==2:
                hole_size = 30
                legthseg = 2.5
            hole_size = (length_-segments*legthseg)/(segments+1)
            safe_dist = (length_-(segments-1)*hole_size-segments*legthseg)/2
                        
            index0=0
            for seg in range(segments):
                number_in_segments=((seg+1)*amount_of_obstacles)//segments-(seg*amount_of_obstacles)//segments
                
                for i in range(number_in_segments):
                    obstacles_array_[index0,0] = safe_dist + (legthseg + hole_size) * seg + i/(number_in_segments-1) * (legthseg) + radie_obstacles_[index0]
                    obstacles_array_[index0,1] = (width_ - width_obs)/2 + width_obs * i/(number_in_segments-1)
                    
                    #print(index0,obstacles_array_[index0,0])
                    
                    index0+=1

            #obstacles_array_ = np.array([[safe_dist+(length_ - 2*safe_dist)*(i)/(amount_of_obstacles-1), width_/2] for i in range(amount_of_obstacles)])


        enviroment = {"width":width_, 
                      "length":length_, 
                      "obstacles":obstacles_array_, 
                      "obstacles_radie":radie_obstacles_, 
                      "borders_y": [0,width_], 
                      "exit": np.array([[0,50],[width_,50]]), 
                      "entery": np.array([[0,50],[width_,50]]), 
                      "flow": np.array([0.5,0.5]), 
                      "intensity":0.5, 
                      "max_agents":200}

        for sim_number in range(tries):
            
            sd = 3

            TT, C, MS = sim(delta_t = dt_, 
                            partflow = partflow_, 
                            people_per_second_per_meter = people_per_second_per_meter_, 
                            peoples = n_people, 
                            y_borders = enviroment["borders_y"], 
                            obstacles = enviroment["obstacles"], 
                            obstacles_radie = enviroment["obstacles_radie"], 
                            radie_person = 0.31/2, 
                            radie_extra=0.25, 
                            length = enviroment["length"], 
                            safe_dist = sd, 
                            plots_varible = plots_varible, 
                            mean_speed_ = 1.34, 
                            std_speed_ = stds[obs_index] * np.ones((n_people)),
                            N_skip = N_skip_animation,
                            max_distance = md)
            
            people_number = np.arange(n_people*5//10, n_people*9//10)

            indices = np.random.choice(people_number, size=3, replace=False)
            times_[obs_index, sim_number, :] = TT
            class_[obs_index, sim_number, :] = C
            max_speed_[obs_index, sim_number, :] = MS
            avg_times[obs_index, sim_number] = np.mean(TT[people_number])
            avg_std[obs_index, sim_number] = np.std(TT[people_number])

            print(f'Simulation {sim_number}, Average time travel: {np.mean(TT[people_number]):.2f} s, Std: {np.std(TT[people_number]):.2f} s, Sample times: {TT[indices]}, classification_all: {C[indices]}, speed: {(length_)/TT[indices]/MS[indices]}  ')
        #dumpa({"obst":obst.tolist(), "avg_times":avg_times.tolist(), "avg_std":avg_std.tolist()},"dataobsall_v1_onedirection.json")
        dumpa(data = {"desc": "a describtion", 
                    "width_obs": width_obs, 
                    "dt": dt_, 
                    "partflow": partflow_, 
                    "people_per_second_per_meter": people_per_second_per_meter_, 
                    "stds": stds.tolist(),
                    "obst": obst.tolist(),
                    "MS": times_.tolist(), 
                    "TT": class_.tolist(), 
                    "C": max_speed_.tolist()},
                filename = filename + ".json", 
                make_a_dump = filename != "noname")

    #print(desired_direction(np.random.random(size=(4,2)), np.zeros((4,2))))s