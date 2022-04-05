# %%
import os
import networkx as nx
import collections
from sql_metadata import Parser
from matplotlib import pyplot as plt
import numpy as np
# %%
def extract_query_join(query:list,talbe_shortcuts:dict):
    joins = []
    for line in query:
        if("=" in line):
            predicates = line.replace("AND","").strip().split("=")
            if("." in predicates[0] and "." in predicates[1]):
                left = predicates[0].split(".")[0].strip()
                right = predicates[1].split(".")[0].strip()
                print(left,right)
                joins.append(sorted([talbe_shortcuts[left],talbe_shortcuts[right]]))
    return joins

def table_shortcut(query_string:str):
    parser = Parser(query_string)
    table_shortcuts = parser.tables_aliases
    return table_shortcuts
    

def extract_file_join(sql_dir):
    sql_files = os.listdir(sql_dir)
    joins = []
    for file in sql_files:
        print(file)
        with open(os.path.join(sql_dir,file),"r") as f:
            query_string = f.read()
            table_shortcuts = table_shortcut(query_string)
        with open(os.path.join(sql_dir,file),"r") as f:
            query = f.readlines()
            join = extract_query_join(query,table_shortcuts)
            joins.extend(join)
        # break
    return joins


# %%
sql_dir = "../data/join-order-benchmark"
joins = extract_file_join(sql_dir)
join_count = collections.Counter(map(tuple,joins))
# %%%
G=nx.Graph()
for k,v in dict(join_count).items():
    G.add_edge(k[0],k[1],weight=v)
# %%
widths = nx.get_edge_attributes(G, 'weight')
nodelist = G.nodes()

plt.figure(figsize=(26,18))

# pos = nx.shell_layout(G)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G,pos,
                       nodelist=nodelist,
                       node_size=3500,
                       node_color='blue',
                       alpha=0.5)
nx.draw_networkx_edges(G,pos,
                       edgelist = widths.keys(),
                       width=np.array(list(widths.values()))/10,
                       edge_color='black',
                       alpha=0.7)
nx.draw_networkx_labels(G, pos=pos,
                        labels=dict(zip(nodelist,nodelist)),
                        font_color='black')
nx.draw_networkx_edge_labels(G,pos,
                             edge_labels=widths,
                             font_color='red')
# plt.show()
plt.savefig("../result/JOB-query-joins.png")

#%%

def extract_contained_tables(query_string):
    parser = Parser(query_string)
    table_shortcuts = parser.tables_aliases
    return table_shortcuts.values()

# sql_dir = "../data/join-order-benchmark"
sql_dir = "../data/bao_sample_queries"

sql_files = os.listdir(sql_dir)
tables_in_query = {}
for file in sql_files:
    with open(os.path.join(sql_dir,file),"r") as f:
        query_string = f.read()
        contained_tables = sorted(extract_contained_tables(query_string))
        tables_in_query[file] = contained_tables
# %%
reversed_dict = {}
for k,v in tables_in_query.items():
    table_string = " ,".join(v)
    if(table_string not in reversed_dict.keys()):
        reversed_dict[table_string] = [k]
    else:
        reversed_dict[table_string].append(k)
# # %%
# ('company_name', 'movie_companies'): 74,
# ('movie_companies', 'title'): 81,
# ('cast_info', 'name'): 54,
# ('movie_info', 'title'): 57,
# ('movie_keyword', 'title'): 77,
# ('keyword', 'movie_keyword'): 75,
# ('cast_info', 'title'): 57,