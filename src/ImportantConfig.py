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