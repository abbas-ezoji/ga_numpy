###############################################################################
'''                     Author: Abbas Ezoji
                     Email: Abbas.ezoji@gmail.com
'''
###############################################################################
import pandas as pd
import numpy as np
import numpy_indexed as npi
import uuid
import random
import pyodbc
from ga_numpy import GeneticAlgorithm as ga
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine
from time import gmtime, strftime

###############################################################################
'''                             parameters                            '''
###############################################################################
city = 36 
start_time = 420
end_time = 1440 
days = 1

population_size = 50
generations = 500

coh_pnlty = 10

coh_fultm = 0.6
coh_lntm  = 0.1
coh_cnt   = 0.05
coh_dffRqTime  = 0.075
coh_dffVisTime  = 0.075

coh_rate = 1

###############################################################################
'''                  Cost calculation functions                             '''
###############################################################################
def cost_fulltime(individual, end_plan):      
    cost = np.abs(end_plan  - end_time) / (end_time - start_time)
      
    return cost

def cost_lentime(individual, all_dist, all_duration):         
    cost = all_dist / (all_duration + all_dist)
      
    return cost
	
def cost_count(individual, meta_data):
    plan = individual
    len_pln = len(plan)
    len_points = len(meta_data)
    cost = np.abs(len_accpt_points - len_pln) / len_points
    
    return cost

def cost_diffTime(individual):
    plan = individual
    max_rqTime = np.max(plan[:,7])     
    max_visTime = np.max(plan[:,6]) - np.min(plan[:,5])
            
    rq_time  = np.apply_along_axis(apply_rqTime, 1, plan)    
    vis_time = np.apply_along_axis(apply_visTime, 1, plan)
    
    plan[:,8] = rq_time
    plan[:,9] = vis_time

    cost_vis_time = np.sum(vis_time) / max_rqTime
    cost_rq_time = np.sum(rq_time) / max_visTime   
    
    return cost_vis_time, cost_rq_time

def cost_rate(individual, meta_data):
    plan = individual
    max_rate = np.max(meta_data[:,10])
    max_rate_plan = np.max(plan[:,10])
    min_rate_plan = np.min(plan[:,10])
    diff_rate = max_rate_plan - min_rate_plan
    len_pln = len(plan)
    chebi_cost = (max_rate - max_rate_plan)/max_rate
    norm_cost = np.sum(max_rate_plan - plan[:,10])/(len_pln*diff_rate)
    cost = np.mean([chebi_cost,norm_cost])
    
    return cost

###############################################################################
'''               individual General functions                          '''
###############################################################################
def set_const(individual, const):
    plan = individual
    for c in const:    
        msk1 = plan[:,3]>=c[5]    
        p = plan[msk1]
        msk2 = p[:,3]<=c[6] 
        p = p[msk2] if c[6]>0 else p
        if len(p)>0:
            min_p = np.min(p[:,3]) 
            p = p[p[:,3]==min_p]
            c[3] = p[0,3]
            plan[plan[:,3]==p[:,3]] = c
        else:
            nearest_dist = np.min(np.abs(plan[:,3] - c[5]))
            msk = np.abs(plan[:,3] - c[5])==nearest_dist
            p = plan[msk]
#            print(p)
            if len(p)>0:
                min_p = np.min(p[:,3]) 
                p = p[p[:,3]==min_p]
                c[3] = p[0,3]
                plan[plan[:,3]==p[:,3]] = c        
        
    plan = plan[plan[:,3].argsort()]
        
    return plan
    
def calc_starttime(individual):
    plan = individual 
    pln_pnt = plan[:,0]
    for i,dist in enumerate(pln_pnt): 
        if i==0: 
            plan[i,3] = start_time  
        elif plan[i-1,2] == 0 and plan[i,2] == 0: # if last and current type not const
            plan[i,4] = dist_mat.loc[pln_pnt[i-1], pln_pnt[i]]
            plan[i,3] = plan[i,4] + plan[i-1,3] + plan[i-1,1]
        elif plan[i-1,2] > 0 or plan[i,2] > 0:
            plan[i,3] = plan[i-1,3] + plan[i-1,1]
           
    return plan

def apply_visTime(a):
    start = a[3]
    end = a[3] + a[1]
    vis_time  = a[5] - start if a[5]>start else 0
    if a[6]!=0:
        vis_time += end - a[6]   if end>a[6] else 0    
    
    return vis_time

def apply_rqTime(a):    
    
    return np.abs(a[1]-a[7])

###############################################################################
'''                  Cost fitness totla function                            '''
###############################################################################
def fitness(individual, meta_data):    
    _, individual = npi.group_by(individual[:,0]).max(individual)
    
#    individual = set_const(individual, const)
    calc_starttime(individual)
    individual = set_const(individual, const)
    calc_starttime(individual)
    
    len_pln = len(individual)
    edge = len_pln - 1   
    pln_pnt = individual[:,0]
    len_points = len(points)
    all_duration = np.sum(individual[:,1])    
    end_plan = individual[edge,3]+individual[edge,1]
    all_dist = end_plan  - all_duration
    
    cost_fultm = cost_fulltime(individual, end_plan)
    cost_lntm  = cost_lentime(individual, all_dist, all_duration)
    cost_cnt   = cost_count(individual, meta_data)
    cost_vis_time, cost_rq_time = cost_diffTime(individual)
    cost_rte   = cost_rate(individual, meta_data)
#    print('cost_fultm: '+str(cost_fultm))
#    print('cost_lntm: '+str(cost_lntm))
#    print('cost_cnt: '+str(cost_cnt))
#    print('cost_diff_rqTime: '+str(cost_diff_rqTime))   
    cost =((coh_fultm*cost_fultm) + 
               (coh_lntm*cost_lntm) + 
               (coh_cnt*cost_cnt) + 
               (coh_dffRqTime*cost_rq_time)+
               (coh_dffVisTime*cost_vis_time)               
              )    
#    print(cost)

    penalty = (coh_rate*cost_rte)/days
    
    return cost *(1 + (coh_pnlty*penalty))

###############################################################################
'''                             connection config                           '''
###############################################################################
USER = 'sa'
PASSWORD = '1qaz!QAZ'
HOST = '192.168.1.36\MSSQL2017'
PORT = '1433'
NAME = 'planning'
engine = create_engine('mssql+pyodbc://{}:{}@{}/{}?driver=SQL+Server' \
                       .format(USER,
                               PASSWORD,
                               HOST,                    
                               NAME
                               ))

###############################################################################
'''                             Fetch data from db                          '''
###############################################################################
df = pd.read_sql_query('''SELECT * FROM 
                          [planning]..plan_attractions WHERE type=0''',
                       con=engine)
df = df.drop(['fullTitle', 'address', 'description','image'], axis=1)

df_city = df[df['city_id']==city]
df_city['rate'] = (df_city['like_no']*60) + (df_city['view_no']*40)

dist_mat_query = ''' SELECT 
                         origin_id as orgin
                        ,destination_id as dist
                        ,len_time as len
                     FROM 
                       plan_distance_mat
                     WHERE
                       origin_id in 
                       (SELECT id FROM plan_attractions
                        WHERE city_id = {0} AND type=0)
                       '''.format(city)
###############################################################################
'''                  Create dist_mat, Const and meta_data                   '''
#################''' Create distance matrix '''################################             
             
dist_df = pd.read_sql_query(dist_mat_query
                          ,con=engine)

dist_mat = pd.pivot_table(dist_df,                           
                          index=['orgin'],
                          columns=['dist'], 
                          values='len', 
                          aggfunc=np.sum)
######################''' Create Costraints '''################################             
                                        
const_df = pd.read_sql_query('SELECT * FROM plan_attractions WHERE type>0',
                             con=engine)

vst_time_from = np.array(const_df['vis_time_from'])
vst_time_to = np.array(const_df['vis_time_to'])
points = np.array(const_df['id'])
rq_time = np.array(const_df['rq_time'])
types = np.array(const_df['type'])
len_points = len(points)
rq_time_mean = np.min(rq_time)

const = np.array([points, 
                      rq_time, 
                      types, 
                      np.zeros(len_points),     # as strat time
                      np.zeros(len_points),     # as distance time
                      np.array(vst_time_from),  # as vst_time_from
                      np.array(vst_time_to),    # as vst_time_to
                      np.array(rq_time),        # as rq_time
                      np.zeros(len_points),     # as diff_rqTime
                      np.zeros(len_points),     # as diff_visTime
                      np.zeros(len_points),     # as rate
                      ],
                      dtype=int).T
                  
len_const = len(const) 
tot_lenTimeConst = np.mean(const[:,1]) * len_const

#########''' Create all accepted Points as meta_data '''#######################                         
plan = []
train_time = gmtime()
time_str = str(train_time.tm_year) + '-' + \
           str(train_time.tm_mon) + '-' + \
           str(train_time.tm_mday) + '-' + \
           str(train_time.tm_hour) + '-' + \
           str(train_time.tm_min) + '-' + \
           str(train_time.tm_sec)
           
present_id = str(city) + '-' + str(days) + '-'  + time_str

for day in range(1,days+1):    

    # day = 1
    last_pints = pd.read_sql_query('''SELECT pd.point_id 
                                    FROM 
                                        plan_plan p
                                        JOIN plan_plan_details pd 
                                        ON pd.plan_id = p."id"
                                    WHERE p.present_id={}
                                    '''.format("'"+str(present_id)+"'"),
                             con=engine)
    
    mask = ~df_city['id'].isin(last_pints['point_id'])
    df_city = df_city[mask]
    
    points = np.array(df_city['id'])
    len_points = len(points)
    vst_time_from = np.array(df_city['vis_time_from'])
    vst_time_to = np.array(df_city['vis_time_to'])
    rq_time = np.array(df_city['rq_time'])  
    rate = np.array(df_city['rate'])   
    rq_time_mean = np.mean(rq_time)
    
    meta_data = np.array([points, 
                          rq_time, 
                          np.zeros(len_points),     # as zero as type
                          np.zeros(len_points),     # as strat time
                          np.zeros(len_points),     # as distance time
                          np.array(vst_time_from),  # as vst_time_from
                          np.array(vst_time_to),    # as vst_time_to
                          np.array(rq_time),        # as rq_time
                          np.zeros(len_points),     # as diff_rqTime
                          np.zeros(len_points),     # as diff_visTime
                          np.array(rate),           # as rate
                          ],
                          dtype=int).T
    
    
    
    tot_lenTime = end_time - start_time
    len_accpt_points = (tot_lenTime-tot_lenTimeConst)/rq_time_mean                      
    
                    ###################################################
    '''                  Create sample gene from meta_data                  '''
                    ###################################################
    
    pln_gene1 = meta_data
    np.random.shuffle(pln_gene1)
    
                    ###################################################
    '''                  Set parameters and Call GA                         '''
                    ###################################################
    if (day==1):
        ga = ga(seed_data=pln_gene1,
                meta_data=meta_data,    
                population_size=population_size,
                generations=generations,
                current_generation_count = 0,
                crossover_probability=0.8,
                mutation_probability=0.2,
                elitism=True,
                by_parent=False,
                maximise_fitness=False)	
        ga.fitness_function = fitness
        
    ga.run()   
    
                    ###################################################
    '''                  Get GA outputs and calculate all cost and 
                         other output featurs      '''
                    ###################################################
    sol_fitness, sol_df = ga.best_individual()
    
    calc_starttime(sol_df)
    individual = set_const(sol_df, const)
    calc_starttime(sol_df)
    
    len_pln = len(sol_df)
    edge = len_pln - 1   
    pln_pnt = sol_df[:,0]
    len_points = len(points)
    all_duration = np.sum(sol_df[:,1])    
    end_plan = sol_df[edge,3]+sol_df[edge,1]
    all_dist = end_plan  - all_duration
        
    cost_fultm = cost_fulltime(sol_df, end_plan)
    cost_lntm  = cost_lentime(sol_df, all_dist, all_duration)
    cost_cnt   = cost_count(sol_df, meta_data)
    cost_vis_time, cost_rq_time = cost_diffTime(sol_df)
    cost_rte   = cost_rate(individual, meta_data)

    diff_full_time = end_plan - end_time
    
    cost =((coh_fultm*cost_fultm) + 
               (coh_lntm*cost_lntm) + 
               (coh_cnt*cost_cnt) + 
               (coh_dffRqTime*cost_rq_time)+
               (coh_dffVisTime*cost_vis_time)+
               (coh_rate*cost_rte)
               )    
    #    print(cost)
    
                    ###################################################
    '''                  Create query for inser plan in db                  '''
                    ###################################################  
    tags = 'test'
    comment = 'test'
    
    query_plan = '''insert into plan_plan (city_id,
    									   present_id,
    									   "coh_fullTime",
    									   "coh_lengthTime",
    									   "coh_countPoints",
    									   "coh_minRqTime",
    									   "cost_fullTime",
    									   "cost_lengthTime",
    									   "cost_countPoints",
    									   "cost_minRqTime",
                                           "cost_rate",
    									   start_time,
    									   end_time,
    									   dist_len,
    									   points_len,
    									   duration_len,
    									   tags,
    									   comment,
                                           day)
                     values ({0}, {1}, 
                             {2}, {3}, {4}, {5},
                             {6}, {7}, {8}, {9},{10},
                             {11}, {12},
                             {13}, {14}, {15}, 
                             {16}, {17}, {18}) 
                   '''.format(city, "'"+str(present_id)+"'",
                              coh_fultm, coh_lntm, coh_cnt, coh_dffRqTime, 
                              cost_fultm, cost_lntm, cost_cnt, cost_rq_time,cost_rte,
                              start_time, end_time,
                              all_dist, len_pln, all_duration,
                              "'"+str(tags)+"'", "'"+str(comment)+"'",
                              day
                              )
    
    engine.execute(query_plan)               
    
    inserted_plan = pd.read_sql_query('''SELECT * 
                                      FROM plan_plan
                                      WHERE present_id = {0} and day = {1}
                                          '''.format( "'"+str(present_id)+"'", day)
                                         ,con=engine)
    plan_id = int(inserted_plan['id'])
    
    for i, sol in enumerate(sol_df):
        qry = '''insert into 
                 plan_plan_details(plan_id, 
                                   "order",
                                   len_time,                              
                                   point_id,
                                   from_time,
                                   dist_to)
                 values({0}, {1}, {2}, {3}, {4}, {5})
                 '''.format(plan_id, i, sol[1], sol[0], sol[3], sol[4])
        engine.execute(qry)    


last_gens = ga.last_generation()   

last_genes = [gene.get_geneInfo() for gene in last_gens]
