import json
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(33)#101)

#datafromobs, dataobsall_v2, "variance_of_speed//variance_of_speed_v2.json"
def load_json(filename="variance_of_speed//data_std_of_speed_0d25.json"):
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

all_mean2_ps = []
all_mean_ps = []
all_mean_psg = []

all_std_ps = []
all_std_psg = []
stds_=np.array(data["stds"])

samples_array = np.array([10])

mean_ms_2 = np.mean(np.abs(max_speed),axis=2)
mean_ms_1 = np.mean(mean_ms_2,axis=1)
mean_ms_0 = np.mean(mean_ms_1,axis=0)
print(max_speed.shape)


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

    std_ps_1 = np.std(mean_ps_2,axis=1)
    std_psg_1 = np.std(partspeed_global.reshape(partspeed_global.shape[0], -1),axis=1)



    meantime2 = np.mean(times_a,axis=2)
    std2 = np.std(times_a,axis=2)

    meantime1 = np.mean(meantime2,axis=1)
    std1 = np.std(meantime2,axis=1)

    obst2 = np.repeat(obst[:, None], times_a.shape[1], axis=1)
    obst3 = np.repeat(obst[:, None, None], times_a.shape[1]*times_a.shape[2], axis=1)
    all_mean2_ps.append(mean_ps_2)
    all_mean_ps.append(mean_ps_1)
    all_std_ps.append(std_ps_1)

    all_mean_psg.append(mean_psg_1)
    all_std_psg.append(std_psg_1)

all_mean2_ps = np.array(all_mean2_ps)
all_mean_ps = np.array(all_mean_ps)
all_std_ps = np.array(all_std_ps)

all_mean_psg = np.array(all_mean_psg)
all_std_psg = np.array(all_std_psg)

#plt.scatter(obst2.flatten(), meantime2.flatten())

#plt.scatter(obst, meantime1)
#plt.errorbar(obst, meantime1, std1)

true_array = np.zeros(samples_array.size,dtype=bool)
true_array[0] = True
musk=10
musk1=5
#c2, m2 = np.polyfit(stds_[:musk1], all_mean_ps[0,:musk1], 1)

c0, m0 = np.polyfit(stds_[:musk1+1], all_mean_ps[0,:musk1+1], 1)
c1, m1 = np.polyfit(stds_[musk1:musk], all_mean_ps[0,musk1:musk], 1)

def f(x, a, b, c,d):
    return b*np.exp(a*x**4)+c-d*x
def f2(x, a, b):
    return b*np.exp(a*x**2)

import scipy.optimize as sp
p1, _ = sp.curve_fit(f,np.ndarray.flatten(np.repeat(stds_[:, np.newaxis], 5, axis=1)), np.ndarray.flatten(all_mean2_ps[0,:,:]), p0=[-16,0.93,0,-0.1],maxfev=10000)
p2, _ = sp.curve_fit(f2,np.ndarray.flatten(np.repeat(stds_[:, np.newaxis], 5, axis=1)), np.ndarray.flatten(all_mean2_ps[0,:,:]), p0=[-16,0.93],maxfev=10000)

#p, _ = sp.curve_fit(f, stds_, all_mean_ps[0,:], p0=[-1.6,0.93,0,0.1])
#print(p)
print(p1)
print(c0,m0)
#print(c2,m2)

#(c1,m1)

#y_exp = f(stds_,p[0],p[1],p[2],p[3])
y1_exp = f(stds_,p1[0],p1[1],p1[2],p1[3])
y2_exp = f2(stds_,p2[0],p2[1])

fig, ax = plt.subplots(1,1,figsize=(5,3))
for k, samples in enumerate(samples_array):
    if true_array[k]:
        ax.errorbar(stds_[:musk], all_mean_ps[k,:musk], all_std_ps[k,:musk],alpha=1, color="tab:red",fmt='o', capsize=5, capthick=1.5,zorder=3)
        #ax.errorbar(stds_[:musk], all_mean_psg[k,:musk], all_std_psg[k,:musk],alpha=0.5, color="tab:green",fmt='o', capsize=6, capthick=2)
        ax.plot(stds_[:musk], m0+stds_[:musk]*c0, '--', color='tab:blue', alpha=1, label='linear-fit $\mathrm{\sigma_v\leq0.2}$', linewidth=2,zorder=2)
        ax.plot(stds_[:musk], m1+stds_[:musk]*c1, "--", color='tab:orange', alpha=1, label='linear-fit $\mathrm{\sigma_v\geq0.2}$', linewidth=2,zorder=1)
        
        #ax.plot(stds_, y1_exp, '--', color='purple', alpha=1, label='fit non-linear', linewidth=2)

        #ax.plot(stds_, y2_exp, '--', color='black', alpha=1, label='fit exponential', linewidth=2)
        ax.scatter(stds_[:musk], all_mean_ps[k,:musk], s=50, color='tab:red', label=f"data",zorder=1)
        #ax.scatter(stds_[:musk], all_mean_psg[k,:musk], s=50, color='tab:green', label=f"Sampled data")

ax.set_ylim(0.75,0.99)

ax.set_ylabel("Efficiency", fontsize=11)# ($\mathrm{\dfrac{<t_{desired}>}{<t_{sim}>}})$")# ($\mathrm{\dfrac{length}{<t_{sim}> <|v_{max}|>}}$)")
ax.set_xlabel("Standard deviation in desired speed, $\mathrm{\sigma_v}$  [$\mathrm{\dfrac{m}{s}}$]", fontsize=11)
ax.legend(loc="lower left", fontsize=11)
fig.tight_layout()
plt.savefig("C://Users//46738//OneDrive//Dokument//VS-program chalmers//SOCS_project//variance_of_speed//figure_std_speed.png", dpi=300, bbox_inches="tight")
plt.show()