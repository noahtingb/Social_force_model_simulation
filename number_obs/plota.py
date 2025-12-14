import json
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(101)

#datafromobs, dataobsall_v2
def load_json(filename="number_obs//dataobsall_v2.json"):
    with open(filename, "r") as f:
        return json.load(f)
    
data = load_json()
#obst = np.array(data["obst"])
#avg_times = np.array(data["avg_times"])
#avg_std = np.array(data["avg_std"])

max_speed = np.array(data["C"])
obst = np.array(data["obst"])
times = np.array(data["MS"])
classification = np.array(data["TT"])
arr1000 = np.arange(times.shape[2], dtype=int)
all_mean_ps = []
all_mean_psg = []

all_std_ps = []
all_std_psg = []

samples_array = np.array([1,3,5,10,20,50,100,400])

mean_ms_2 = np.mean(np.abs(max_speed),axis=2)
mean_ms_1 = np.mean(mean_ms_2,axis=1)
mean_ms_0 = np.mean(mean_ms_1,axis=0)
print(mean_ms_0)
print(1.3*1.3*3.6)


for samples in samples_array:

    choises = np.zeros((times.shape[0], times.shape[1], samples),dtype=int)
    times_a =  np.zeros(choises.shape)
    max_speed_a =  np.zeros(choises.shape)
    partspeed =  np.zeros(choises.shape)
    for i in range(times.shape[0]):
        for j in range(times.shape[1]):
            choises[i,j,:] = np.random.choice(arr1000[400:800], samples, replace=False)
            times_a[i,j,:] = times[i,j,choises[i,j,:]]
            max_speed_a[i,j,:] = max_speed[i,j, choises[i,j,:]]

    partspeed = (51/times_a)/np.abs(max_speed_a)
    partspeed_global = (51/times_a)/mean_ms_0
    

    mean_ps_2 = np.mean(partspeed,axis=2)
    mean_psg_2 = np.mean(partspeed_global,axis=2)

    #std_ps_2 = np.std(partspeed,axis=2)

    mean_ps_1 = np.mean(mean_ps_2,axis=1)
    mean_psg_1 = np.mean(mean_psg_2,axis=1)

    std_ps_1 = np.std(partspeed.reshape(partspeed.shape[0], -1),axis=1)
    std_psg_1 = np.std(partspeed_global.reshape(partspeed_global.shape[0], -1),axis=1)



    meantime2 = np.mean(times_a,axis=2)
    std2 = np.std(times_a,axis=2)

    meantime1 = np.mean(meantime2,axis=1)
    std1 = np.std(meantime2,axis=1)

    obst2 = np.repeat(obst[:, None], times_a.shape[1], axis=1)
    obst3 = np.repeat(obst[:, None, None], times_a.shape[1]*times_a.shape[2], axis=1)
    all_mean_ps.append(mean_ps_1)
    all_std_ps.append(std_ps_1)

    all_mean_psg.append(mean_psg_1)
    all_std_psg.append(std_psg_1)

all_mean_ps = np.array(all_mean_ps)
all_std_ps = np.array(all_std_ps)

all_mean_psg = np.array(all_mean_psg)
all_std_psg = np.array(all_std_psg)

#plt.scatter(obst2.flatten(), meantime2.flatten())

#plt.scatter(obst, meantime1)
#plt.errorbar(obst, meantime1, std1)
print(obst, all_mean_ps.shape)
true_array = np.zeros(samples_array.size,dtype=bool)
true_array[np.array([3])] = True

c0, m0 = np.polyfit(obst, all_mean_ps[1,:], 1)
c1, m1 = np.polyfit(obst, all_mean_ps[1,:], 1)
c2, m2 = np.polyfit(obst, all_mean_ps[2,:], 1)
c3, m3 = np.polyfit(obst, all_mean_ps[3,:], 1)
c20, m20 = np.polyfit(obst, all_mean_ps[4,:], 1)
c50, m50 = np.polyfit(obst, all_mean_ps[5,:], 1)
c100, m100 = np.polyfit(obst, all_mean_ps[6,:], 1)
c400, m400 = np.polyfit(obst, all_mean_ps[7,:], 1)
print(m3,c3)
fig, ax = plt.subplots(1,1,figsize=(6,3))
for k, samples in enumerate(samples_array):
    if true_array[k]:
        #plt.errorbar(obst, all_mean_ps[k,:], all_std_ps[k,:],alpha=0.5, fmt='o', capsize=6, capthick=2)
        ax.plot(obst, m1+obst*c1, '--', color='tab:red', alpha=0.7, label=f'Linear fit: 3 samples per sim', linewidth=2)
        ax.plot(obst, m3+obst*c3, '--', color='tab:green',alpha=0.7, label=f'Linear fit: 10 samples per sim', linewidth=2)
        ax.plot(obst, m100+obst*c100, '--', color='tab:orange', alpha=0.7, label=f'Linear fit: 100 samples per sim', linewidth=2)
        ax.scatter(obst, all_mean_ps[k,:],  color='tab:blue', label=f"10 sample from 10 simulations")

        #plt.scatter(obst, all_mean_psg[k,:], label=f"ss = {samples}")
        #plt.errorbar(obst, all_mean_psg[k,:], all_std_psg[k,:],alpha=0.5, fmt='o', capsize=6, capthick=2)        
ax.set_ylabel("mean propotion of max speed")
ax.set_xlabel("number of obstacles")
ax.legend()
fig.tight_layout()
plt.show()