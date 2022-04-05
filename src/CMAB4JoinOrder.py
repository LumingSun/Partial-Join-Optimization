# %%
from Query import Query, Workload
from ImportantConfig import Config, PGRunner
import random
import os

config = Config()


# %%
sql_dir = "/home/sunluming/join/PartialJoin/data/join-order-benchmark"
workload = Workload(sql_dir,method="occurance_in_plan")

# %%
def get_plan_latency(query):      
    pgrunner = PGRunner(config.dbName,config.userName,config.password,config.ip,config.port)
    latency, plan = pgrunner.execution_cost(query)
    # plan = pgrunner.optimizer_cost(query)
    return latency, plan

# %%
with open("/home/sunluming/join/PartialJoin/data/join-order-benchmark/13a.sql") as f:
    query = f.read().replace("\n"," ")

default_latency, default_plan = get_plan_latency(query)
print("default plan latency: ",default_latency)
Query_q = Query(query,workload)
hints = Query_q.generate_join_order(random.choice(Query_q.candidate_arms))
latency, plan = get_plan_latency(hints+query)

print("selected plan latency: ",latency)
# %%
