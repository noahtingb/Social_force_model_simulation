import json
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(100)

#datafromobs, dataobsall_v2, color=colors[colr_index%colors.size]
def load_json(filename="design_c//data_design_choice.json"):
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

all_mean1_ps = []

all_mean_ps = []
all_mean_psg = []

all_std_ps = []
all_std_psg = []

samples_array = np.array([1,3,5,10,400])

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
            choises[i,j,:] = np.random.choice(arr1000[450:850], samples, replace=False)
            times_a[i,j,:] = times[i,j,choises[i,j,:]]
            max_speed_a[i,j,:] = max_speed[i,j, choises[i,j,:]]

    partspeed = (50/times_a)/np.abs(max_speed_a)
    partspeed_global = (50/times_a)/mean_ms_0

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

    all_mean1_ps.append(mean_ps_2)

    all_mean_ps.append(mean_ps_1)
    all_std_ps.append(std_ps_1)

    all_mean_psg.append(mean_psg_1)
    all_std_psg.append(std_psg_1)


all_mean1_ps = np.array(all_mean1_ps)

all_mean_ps = np.array(all_mean_ps)
all_std_ps = np.array(all_std_ps)

all_mean_psg = np.array(all_mean_psg)
all_std_psg = np.array(all_std_psg)

#plt.scatter(obst2.flatten(), meantime2.flatten())

#plt.scatter(obst, meantime1)
#plt.errorbar(obst, meantime1, std1)

for j, samples in enumerate(samples_array):
    print(f"samples = {samples}")
    print("\n\neffectivity\t14\u00B0\tno angle\t0\u00B0\nmean:\t\t"+"\t".join([f"{i:.3f}" for i in  np.mean(all_mean1_ps[j,:,:],axis=1)])+"\nstd:\t\t"+"\t".join([f"{i:.3f}" for i in  np.std(all_mean1_ps[j,:,:],axis=1)])+"\n\n")
    print("\n\neffectivity\t14\u00B0\tno angle\t0\u00B0\nmean:\t\t"+"\t".join([f"{i:.3f}" for i in  np.mean(all_mean1_ps[j,:,:],axis=1)])+"\nstd:\t\t"+"\t".join([f"{i:.3f}" for i in  np.std(all_mean1_ps[j,:,:],axis=1)])+"\n\n")


#print(obst, all_mean_ps.shape)
true_array = np.zeros(samples_array.size,dtype=bool)
true_array[0] = True

c = [0,0,0]
m = [0,0,0]
c[0], m[0] = np.polyfit(obst, all_mean_ps[0,:], 1)
c[1], m[1] = np.polyfit(obst, all_mean_ps[1,:], 1)
c[2], m[2] = np.polyfit(obst, all_mean_ps[2,:], 1)


fig, ax = plt.subplots(1,1,figsize=(6,3))
for k, samples in enumerate(samples_array):
    if true_array[k]:
        #plt.errorbar(obst, all_mean_ps[k,:], all_std_ps[k,:],alpha=0.5, fmt='o', capsize=6, capthick=2)
        ax.plot(obst, m[k]+obst*c[k], '--', color='tab:red', alpha=0.7, label=f'Linear fit: {samples} samples per sim', linewidth=2)

#        ax.plot(obst, m0+obst*c0, '--', color='tab:green',alpha=0.7, label=f'Linear fit: 10 samples per sim', linewidth=2)
        print(obst.shape,all_mean1_ps.shape,samples)
        ax.scatter(np.repeat(obst[:, np.newaxis], all_mean1_ps.shape[2], axis=1), all_mean1_ps[k,:,:],  color='tab:blue', label=f"10 sample from 10 simulations")

        #plt.scatter(obst, all_mean_psg[k,:], label=f"ss = {samples}")
        #plt.errorbar(obst, all_mean_psg[k,:], all_std_psg[k,:],alpha=0.5, fmt='o', capsize=6, capthick=2)        
ax.set_ylabel("mean propotion of max speed")
ax.set_xlabel("number of obstacles")
ax.legend()
fig.tight_layout()

print("all_mean_ps",all_mean_ps,"\n",all_std_ps)

plt.show()