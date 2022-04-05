# %%
import json
import os
import psycopg2
from ImportantConfig import Config, PGRunner

config = Config()

# %%
def extract_join_order(tree):
    queue = []
    stack = []
    queue.append(tree["Plan"])
    while(len(queue)!=0):
        node = queue.pop(0)
        if("Plans" in node):
            for child in node["Plans"]:
                queue.append(child)
        else:
            stack.append(node["Relation Name"])
    return stack[::-1]
    
# %%

def get_plan(sql_dir,file):
    with open(os.path.join(sql_dir,file),"r") as f:
        query = f.read().replace("\n"," ")            
    pgrunner = PGRunner(config.dbName,config.userName,config.password,config.ip,config.port)
    plan = pgrunner.optimizer_cost(query)
    # print(plan)
    order = extract_join_order(plan)
    # print(order)
    return order
# %%
orders = []
sql_dir = "../data/join-order-benchmark"
sql_files = os.listdir(sql_dir)
for file in sql_files:
    orders.append(get_plan(sql_dir,file))

# %%
orders = []
sql_dir = "../data/join-order-benchmark"
sql_files = os.listdir(sql_dir)
for i in range(33):
    order = []
    for file in sql_files:
        if(file.startswith(str(i))):
            order.append(get_plan(sql_dir,file))
    orders.append(order)




# %%
from apyori import apriori

results = list(apriori(orders,             
                min_support=0.02,
                min_confidence=0.80,
                min_lift=1.0,
                max_length=None))

results = list(apriori(orders))
# %%
for i in range(2,17):
    tmp_list = []
    for each in orders:
        if(len(each)>=i):
            tmp_list.append(each[:i])
    for first in tmp_list:
        for second in tmp_list:
            if(sorted(first)==sorted(second) and first!=second):
                print("first: ",first)
                print("second: ",second)
                
# %%
