import psycopg2

class Config:
    def __init__(self,):
        self.sytheticDir = "Queries/synthetic"
        self.JOBDir = "Queries/JOB"
        self.schemaFile = "schema.sql"
        self.dbName = "job"
        self.userName = "sunluming"
        self.password = ""
        self.ip = "127.0.0.1"
        self.port = 5432

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
    
    def execution_cost(self,query,time_limit,enable_parallel=False):
        cursor = self.con.cursor()
        cursor.execute("SET statement_timeout TO {}".format(time_limit))
        if not enable_parallel:
            cursor.execute("set max_parallel_workers_per_gather = 0")
        cursor.execute("load 'pg_hint_plan'")
        try:
            cursor.execute("EXPLAIN (FORMAT JSON,ANALYSE)" + query + ";")
            rows = cursor.fetchall()
            cursor.close()
            return rows[0][0][0]["Execution Time"], rows[0][0][0]
        except psycopg2.errors.QueryCanceled:
            cursor.close()
            return time_limit, "Empty Plan"

def get_plan_latency(config,query,time_limit=300000,enable_parellel=False):      
    pgrunner = PGRunner(config.dbName,config.userName,config.password,config.ip,config.port)
    latency, plan = pgrunner.execution_cost(query,time_limit,enable_parellel)
    # plan = pgrunner.optimizer_cost(query)
    return latency, plan