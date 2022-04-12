# %%
from Query import Query, Workload
from ImportantConfig import Config, PGRunner,get_plan_latency
import random
import os
from datetime import datetime
import json

config = Config()

def save_experience(file,query,selected_arm,reward,context):
    with open(file,"a") as f:
        f.write(query+" "+str(selected_arm)+" "+str(reward)+" "
                +" ".join(map(str, context))+"\n")
    print("[Success]: Experience logged arm: {}, reward: {}.".format(selected_arm,reward))
        
def save_hint(file,query,selected_arm,reward,hint):
    with open(file,"a") as f:
        f.write(query+" "+str(selected_arm)+" "+str(reward)+" "
                +hint+"\n")

# %%
RANDOM_SIZE = 3
folder_time = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
dataset = "join-order-benchmark"
# dataset = "bao_sample_queries"
sql_dir = "/home/sunluming/join/PartialJoin/data/{}".format(dataset)
log_file = "/home/sunluming/join/PartialJoin/data/log_files/{}-exp-{}.txt".format(dataset,folder_time)
hint_file = "/home/sunluming/join/PartialJoin/data/hint_files/{}-exp-{}.txt".format(dataset,folder_time)
plan_folder = "/home/sunluming/join/PartialJoin/data/plan_data/{}".format(dataset)
workload = Workload(sql_dir,method="occurance_in_plan")
print(workload.top_tables)
# %%
# sql_files = os.listdir(sql_dir)
for file in workload.sql_file:
    print("[INFO]: Processing {}.".format(file))
    with open(os.path.join(sql_dir,file)) as f:
        query = f.read().replace("\n"," ")

    Query_q = Query(query,workload)
    context = Query_q.context
    
    default_latency, default_plan = get_plan_latency(config, query, enable_parellel=True)
    with open(os.path.join(plan_folder,(file.split(".")[0]+"_arm-0.json")),"w") as f:
        json.dump(default_plan, f)    
    print("[INFO]: Default plan latency: ",default_latency)
    save_experience(log_file,file,0,default_latency,context)
    save_hint(hint_file,file,0,default_latency," ")
    print("[INFO]: Arm space: ", Query_q.candidate_arms)
    
    for cnt in range(RANDOM_SIZE):
        random_arm = random.choice(Query_q.candidate_arms)
        print("[INFO]: Selected arm: ",random_arm)
        
        hints = Query_q.generate_join_order(random_arm)
        equal_arm_mappings = Query_q.arm_remapping(random_arm)
        print("[INFO]: Equal arms: ", equal_arm_mappings)
        
        latency, plan = get_plan_latency(config,hints+query,time_limit=2*int(default_latency), enable_parellel=True)
        save_hint(hint_file,file,random_arm,latency,hints)

        print("[INFO]: Selected plan latency: ",latency)
        for arm in equal_arm_mappings:
            save_experience(log_file,file,arm,latency,context)
            with open(os.path.join(plan_folder,(file.split(".")[0]+"_arm-{}.json".format(arm))),"w") as f:
                json.dump(plan, f)    
    # break
    print("")
# %%
