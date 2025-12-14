import json
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(101)

#datafromobs, dataobsall_v2
def load_json(filename="design_c//designchoice_02V_v2.json"):
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

time_limit = 52*3.6


ll=500
pp=1000
number_in_timelimit = np.zeros(times.shape[:-1],dtype=int)
in_timelimit = np.zeros(number_in_timelimit.shape+(pp-ll,))
mean_pass = np.zeros(times.shape[0])
std_pass = np.zeros(times.shape[0])
for i in range(times.shape[0]):
    for j in range(times.shape[1]):
        in_timelimit[i,j,:] = np.where((times[i,j,ll:pp]>0)& (times[i,j,ll:pp]<time_limit),True,False)
        number_in_timelimit[i,j] = np.sum(in_timelimit[i,j,:])
    mean_pass[i]=np.mean(number_in_timelimit[i,:])/(pp-ll)
    std_pass[i] = np.std(number_in_timelimit[i,:])/(pp-ll)

print("passrate\t0\tangle\tno angle\nmean:\t\t","\t".join([f"{i:.4f}" for i in  mean_pass]),"\nstd:\t\t","\t".join([f"{i:.4f}" for i in  std_pass]))
#plt.scatter(obst2.flatten(), meantime2.flatten())

#plt.scatter(obst, meantime1)
#plt.errorbar(obst, meantime1, std1)
fig, ax = plt.subplots(1,1,figsize=(6,3))
for k in range(obst.shape[0]):
    ax.scatter(np.repeat(obst[:, np.newaxis], 10, axis=1),number_in_timelimit,  color='tab:blue', label=f"10 sample from 10 simulations")

        #plt.scatter(obst, all_mean_psg[k,:], label=f"ss = {samples}")
        #plt.errorbar(obst, all_mean_psg[k,:], all_std_psg[k,:],alpha=0.5, fmt='o', capsize=6, capthick=2)        
ax.set_ylabel("mean propotion of max speed")
ax.set_xlabel("number of obstacles")
ax.legend()
fig.tight_layout()
plt.show()