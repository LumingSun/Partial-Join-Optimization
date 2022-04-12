# %%
with open("/home/sunluming/join/PartialJoin/data/log_files/join-order-benchmark-exp-2022-04-10_05-20-33_PM.txt","r") as f:
    data = f.readlines()
    
performance = {}
for line in data:
    line = line.split(" ")
    sql = line[0].split(".")[0]
    if sql not in performance:
        performance[sql] = {line[1]:line[2]}
    else:
        performance[sql][line[1]] = line[2]
# %%
for k,v in performance.items():
    for key,val in v.items():
        if(float(v[key])<float(v['0'])):
            print(k)
            print(key)
# %%
