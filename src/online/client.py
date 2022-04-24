import random
import os
from datetime import datetime
import json
import sys
import train
from model import Model 
# sys.path.append(os.getcwd())
# sys.path.append("..") 
from Query import Query, Workload
from ImportantConfig import Config, get_plan_latency
import storage
#os.chdir(sys.path[0]) 使用文件所在目录
 #添加工作目录到模块搜索目录列表

PG_OPTIMIZER_INDEX = 0
DEFAULT_MODEL_PATH = "deepo_default_model"
TMP_MODEL_PATH = "deepo_tmp_model"
OLD_MODEL_PATH = "deepo_previous_model"
COLLECT_INITIAL = False

if __name__ == "__main__":
    config = Config()
    
    folder_time = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    dataset = "join-order-benchmark"
    sql_dir = "/home/sunluming/join/PartialJoin/data/{}".format(dataset)
    log_file = "/home/sunluming/join/PartialJoin/data/log_files/{}-exp-{}.txt".format(dataset,folder_time)
    hint_file = "/home/sunluming/join/PartialJoin/data/hint_files/{}-exp-{}.txt".format(dataset,folder_time)
    plan_folder = "/home/sunluming/join/PartialJoin/data/plan_data/{}".format(dataset)
    workload = Workload(sql_dir,method="occurance_in_plan")
    


    # save plan without hint for initial model training
    if COLLECT_INITIAL:
        for file in workload.sql_file:
            print("[INFO]: Processing {}.".format(file))
            with open(os.path.join(sql_dir,file)) as f:
                query = f.read().replace("\n"," ")
            Query_q = Query(query,workload)
            context = Query_q.context
            default_latency, default_plan = get_plan_latency(config, query, enable_parellel=True)
            # save raw query
            storage.record_reward(default_plan,float(default_latency),0)
            
        
    model = Model()
    if os.path.exists(DEFAULT_MODEL_PATH):
        print("Loading existing model")
        model.load_model(DEFAULT_MODEL_PATH)
        
    train.train_and_swap(DEFAULT_MODEL_PATH, OLD_MODEL_PATH, TMP_MODEL_PATH, verbose=True)
    
    
    # parse plan and train model
    