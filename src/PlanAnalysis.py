# %%
import json
import os
import psycopg2
from ImportantConfig import Config

config = Config()
# %%
class PGConfig:
    def __init__(self):
        self.keepExecutedPlan =True
        self.maxTimes = 5
        self.maxTime = 300000
class PGRunner:
    def __init__(self,dbname = '',user = '',password = '',host = '',port = ''):
        self.con = psycopg2.connect(database=dbname, user=user,
                               password=password, host=host, port=port)

    def optimizer_cost(self, query):
        query = "EXPLAIN (FORMAT JSON) " + query + ";"
        cursor = self.con.cursor()
        settings = "set max_parallel_workers_per_gather = 0; "
        cursor.execute(settings+query)
        rows = cursor.fetchone()
        cursor.close()
        return rows[0][0]

# %%



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

def get_plan(sql_dir):
    orders = []
    sql_files = os.listdir(sql_dir)
    for file in sql_files:
        print(file)
        with open(os.path.join(sql_dir,file),"r") as f:
            query = f.read().replace("\n"," ")            
        pgrunner = PGRunner(config.dbName,config.userName,config.password,config.ip,config.port)
        plan = pgrunner.optimizer_cost(query)
        # print(plan)
        order = extract_join_order(plan)
        # print(order)
        orders.append(order)
        # break
    return orders
# %%
sql_dir = "../data/join-order-benchmark"
orders = get_plan(sql_dir)
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
