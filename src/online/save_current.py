# %%
import json
import storage

# %%
log_file = "/home/sunluming/join/PartialJoin/data/log_files/join-order-benchmark-exp-2022-04-17_06-56-09_PM.txt"

with open(log_file,"r") as f:
    logs = f.readlines()
    
for log in logs:
    sql = log.split(" ")[0].split(".")[0]
    arm = log.split(" ")[1]
    if(arm!='0'):
         continue
    reward = log.split(" ")[2]
    
    plan_path = "/home/sunluming/join/PartialJoin/data/plan_data/join-order-benchmark/{}_arm-{}.json".format(sql,arm)
    with open(plan_path,"r") as f:
        plan = json.load(f)
    # break
    storage.record_reward(plan,float(reward),int(arm))
    
    
# %%
