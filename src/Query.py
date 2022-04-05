# %%
from sql_metadata import Parser
from itertools import permutations
import os
import collections
from ImportantConfig import Config, PGRunner


def table_shortcut(query_string:str):
    parser = Parser(query_string)
    table_shortcuts = parser.tables_aliases
    return table_shortcuts

def extract_query_join(query:list,talbe_shortcuts:dict,use_shortcuts=True):
    joins = []
    for line in query:
        if("=" in line):
            predicates = line.replace("AND","").strip().split("=")
            if("." in predicates[0] and "." in predicates[1]):
                left = predicates[0].split(".")[0].strip()
                right = predicates[1].split(".")[0].strip()
                # print(left,right)
                if(use_shortcuts):
                    joins.append(sorted([left,right]))
                else:
                    joins.append(sorted([talbe_shortcuts[left],talbe_shortcuts[right]]))
    return joins

def digital_in_shortcut(shortcut):
    for key in shortcut.keys():
        if(key[-1].isdigit()):
            return True
    return False

def extract_join_order(tree,tables_shortcuts):
    queue = []
    stack = []
    queue.append(tree["Plan"])
    while(len(queue)!=0):
        node = queue.pop(0)
        if("Plans" in node):
            for child in node["Plans"]:
                queue.append(child)
        else:
            stack.append(tables_shortcuts[node["Relation Name"]])
    return stack[::-1]
    

def get_plan(sql_dir,file,config,tables_shortcuts):
    with open(os.path.join(sql_dir,file),"r") as f:
        query = f.read().replace("\n"," ")            
    pgrunner = PGRunner(config.dbName,config.userName,config.password,config.ip,config.port)
    plan = pgrunner.optimizer_cost(query)
    # print(plan)
    order = extract_join_order(plan,tables_shortcuts)
    # print(order)
    return order

class Workload():
    def __init__(self,sql_dir,threshold=5,method='occurance_in_query'):
        self.sql_dir = sql_dir
        self.joins = []
        self.join_count = None
        self.workload_size = 0
        self.tables = []
        self.sql_file = []
        self.join_orders = None
        self.threshold = threshold
        self.method = method
        self.top_tables = []
        self.plan_order = []
        self.tables_shortcuts = {}
        self.shortcuts_tables = {}
        self.candidate_arms = [None]
        self.workload_info()
        
    def workload_info(self):
        sql_files = sorted(os.listdir(self.sql_dir))
        for file in sql_files:
            with open(os.path.join(self.sql_dir,file),"r") as f:
                query_string = f.read()
                table_shortcuts = table_shortcut(query_string)
                if(digital_in_shortcut(table_shortcuts)):
                    # print(file)
                    continue
                self.shortcuts_tables = dict(self.shortcuts_tables,**table_shortcuts)
                self.workload_size += 1
                self.sql_file.append(file)
            with open(os.path.join(self.sql_dir,file),"r") as f:
                query = f.readlines()
                join = extract_query_join(query,table_shortcuts)
                self.joins.extend(join)
        self.join_count = collections.Counter(map(tuple,self.joins))
        self.tables_shortcuts = dict((v, k) for k, v in self.shortcuts_tables.items())
        
        if(self.method=="occurance_in_join"):
            for each in self.join_count.most_common():
                if(len(self.top_tables)>=self.threshold):
                    break
                if(each[0][0] not in self.top_tables):
                    self.top_tables.append(each[0][0])
                if(each[0][1] not in self.top_tables):
                    self.top_tables.append(each[0][1])
        elif(self.method=="occurance_in_plan"):
            self.top_tables = self.table_occurance_in_plan()[:self.threshold]
        elif(self.method=="occurance_in_query"):
            self.top_tables = self.table_occurance_in_query()[:self.threshold]
            
        self.candidate_arms.extend(list(permutations(self.top_tables)))
        # self.candidate_arms.append(None)
        
            
            
    def table_occurance_in_query(self,with_occur=False):
        table_occurance = {}
        for file in self.sql_file:
            with open(os.path.join(self.sql_dir,file),"r") as f:
                query_string = f.read()
                tables = table_shortcut(query_string).keys()
                for table in tables:
                    if(table not in table_occurance):
                        table_occurance[table] = 1
                    else:
                        table_occurance[table] += 1
        top_tables = sorted(table_occurance.items() , key=lambda t : t[1],reverse=True)
        if(with_occur):
            return top_tables
        else:
            top_tables = [x[0] for x in top_tables]
            return top_tables
                
    def table_occurance_in_plan(self):
        plan_orders = []
        table_occurance = self.table_occurance_in_query(with_occur=True)
        # print("table order in query occurance: ",table_occurance)
        table_occurance = {each[0]:each[1] for each in table_occurance}
        for file in self.sql_file:
            plan_orders.append(get_plan(self.sql_dir,file,config=Config(),tables_shortcuts=self.tables_shortcuts))
        table_count = {}
        for each in plan_orders:
            for i,table in enumerate(each):
                if(table not in table_count):
                    table_count[table] = len(each)-i
                else:
                    table_count[table] += len(each)-i
        # print("table order in plan order", sorted(table_count.items() , key=lambda t : t[1],reverse=True))
        weight = {}
        for table in table_occurance.keys():
            weight[table] = table_occurance[table]/self.workload_size*table_count[table]
        # print("table order with occurance weight (plan):", sorted(weight.items() , key=lambda t : t[1],reverse=True))
        return [x[0] for x in sorted(weight.items() , key=lambda t : t[1],reverse=True)]

class Query():
    
    def __init__(self,query_string,workload):
        """Query Class

        Args:
            query_string (string): query input
        
        shortcuts_tables (dict): 
            key - table shortcut  
            value - table name
        candidate_arms (list):
            available arms
        """
        self.query_string = query_string
        self.shortcuts_tables = None
        self.candidate_arms = [0]
        self.workload = workload
        self.tables_in_query_arm = None
        self.extract_tables()
        self.generate_arms()
 
    def extract_tables(self):
        self.shortcuts_tables = table_shortcut(self.query_string)
        
    def generate_join_order(self,selected_arm):
        if(selected_arm==0):
            return ""
        else:
            orders = (self.workload.candidate_arms[selected_arm][:len(self.tables_in_query_arm)])
            hints = "/*+ Leading({}) */".format(" ".join(orders))
            return hints
    
    def available_arms(self):
        tables_in_query_arm = []
        tables_in_workload_arm = self.workload.top_tables
        for k,v in self.shortcuts_tables.items():
            if(k in tables_in_workload_arm):
                tables_in_query_arm.append(k)
        t_num = len(tables_in_query_arm)
        self.tables_in_query_arm = sorted(tables_in_query_arm)
        
        for idx,arm in enumerate(self.workload.candidate_arms[1:]):
            if(sorted(arm[:t_num]) == self.tables_in_query_arm):
                self.candidate_arms.append(idx)
                 
    def generate_arms(self,join_order=True,physical_operator=False):
        if(physical_operator==True):
            raise NotImplementedError("physical operator arms not implemented")
        if(join_order==True):
            return self.available_arms()
                
    
query = """
SELECT MIN(mc.note) AS production_note,
       MIN(t.title) AS movie_title,
       MIN(t.production_year) AS movie_year
FROM company_type AS ct,
     info_type AS it,
     movie_companies AS mc,
     movie_info_idx AS mi_idx,
     title AS t
WHERE ct.kind = 'production companies'
  AND it.info = 'bottom 10 rank'
  AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'
  AND t.production_year BETWEEN 2005 AND 2010
  AND ct.id = mc.company_type_id
  AND t.id = mc.movie_id
  AND t.id = mi_idx.movie_id
  AND mc.movie_id = mi_idx.movie_id
  AND it.id = mi_idx.info_type_id;
"""
  

# %%
# sql_dir = "../data/join-order-benchmark"
# sql_dir = "../data/bao_sample_queries"
# w = Workload(sql_dir,method="occurance_in_join")
# w = Workload(sql_dir,method="occurance_in_plan")

# # %%
# q = Query(query,w)
# q.extract_tables()
# q.generate_arms()
# %%

# %%
