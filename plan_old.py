###############################################################################
'''                     Author: Abbas Ezoji
                     Email: Abbas.ezoji@gmail.com
                               '''
###############################################################################
import pandas as pd
import numpy as np
from ga_numpy import GeneticAlgorithm as ga
import numpy_indexed as npi
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine
import uuid
import random


###############################################################################
'''                             parameters                            '''
###############################################################################

city = 36 
start_time = 480
end_time = 1440

coh_fultm = 0.6
coh_lntm  = 0.2
coh_cnt   = 0.05
coh_dffRqTime  = 0.075
coh_dffVisTime  = 0.075

coh_pnlty = 0

#constraints = [[420,  540,  45, 1],    # breakfast time
#               [720,  960,  60, 2],    # lunch time
#               [1200, 1320, 50, 3],    # dinner time
#               [1380, 1440, 60, 4]]    # night sleep time
        
###############################################################################
'''                             connection config                           '''
###############################################################################
USER = 'planuser'
PASSWORD = '1qaz!QAZ'
HOST = 'localhost'
PORT = '5432'
NAME = 'planning'
db_connection = "postgresql://{}:{}@{}:{}/{}".format(USER,
                                                     PASSWORD,
                                                     HOST,
                                                     PORT,
                                                     NAME
                                                        )
engine = create_engine(db_connection)

###############################################################################
'''                             Fetch data from db                          '''
###############################################################################
df = pd.read_sql_query('SELECT * FROM	plan_attractions',con=engine)
df = df.drop(['image'], axis=1)

df_city = df[df['city_id']==city]

dist_mat_query = ''' select 
                         origin_id as orgin
                        ,destination_id as dist
                        ,len_time as len
                     from 
                       plan_distance_mat
                     where
                       origin_id in 
                       (select id from plan_attractions
                        where city_id = {0} and type=0)
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
                                        
const_df = df_city[df_city['type']>0]

vst_time_from = np.array(const_df['vis_time_from'])
vst_time_to = np.array(const_df['vis_time_to'])
points = np.array(const_df['id'])
rq_time = np.array(const_df['rq_time'])
types = np.array(const_df['type'])
len_points = len(points)

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
                      ],
                      dtype=int).T
                  
len_const = len(const)
tot_lenTimeConst = np.max(const[:,1]) * len_const

#########''' Create all accepted Points as meta_data '''#######################             
             
df_city = df_city[df_city['type']==0]

vst_time_from = np.array(df_city['vis_time_from'])
vst_time_to = np.array(df_city['vis_time_to'])
points = np.array(df_city['id'])
rq_time = np.array(df_city['rq_time'])
types = np.array(df_city['type'])
len_points = len(points)
rq_time_mean = np.min(rq_time)

len_accpt_points = (end_time - start_time)/rq_time_mean


meta_data = np.array([points, 
                      rq_time, 
                      types, 
                      np.zeros(len_points),     # as strat time
                      np.zeros(len_points),     # as distance time
                      np.array(vst_time_from),  # as vst_time_from
                      np.array(vst_time_to),    # as vst_time_to
                      np.array(rq_time),        # as rq_time
                      np.zeros(len_points),     # as diff_rqTime
                      np.zeros(len_points),     # as diff_visTime
                      ],
                      dtype=int).T
                      

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
        elif plan[i-1,2] == 0 and plan[i,2] == 0:
            plan[i,4] = dist_mat.loc[pln_pnt[i-1], pln_pnt[i]]
            plan[i,3] = plan[i,4] + plan[i-1,3] + plan[i-1,1]
        elif plan[i-1,2] == 1 or plan[i,2] == 1:
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
'''                  Create sample gene from meta_data                      '''
###############################################################################

pln_gene1 = meta_data
np.random.shuffle(pln_gene1)

###############################################################################
'''                  Cost calculation functions                             '''
###############################################################################
def cost_fulltime(individual, end_plan):      
    cost = np.abs(end_plan  - end_time) / 1440.0     
      
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

###############################################################################
'''                  Cost fitness totla function                            '''
###############################################################################
def fitness(individual, meta_data):    
    _, individual = npi.group_by(individual[:,0]).max(individual)
    
#    individual = set_const(individual, const)
    calc_starttime(individual)
    individual = set_const(individual, const)
    
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
#    msk = np.isin(const[:,0], individual[:,0])
#    notUsed_const = const[~msk]
#    penalty = np.sum(notUsed_const[:,1]) / tot_lenTimeConst   
    
    return cost #*(1 + (coh_pnlty*penalty))

###############################################################################
'''                  Set parameters and Call GA                             '''
###############################################################################
ga = ga(seed_data=pln_gene1,
        meta_data=meta_data,    
        population_size=50,
        generations=200,
        crossover_probability=0.8,
        mutation_probability=0.2,
        elitism=True,
        by_parent=False,
        maximise_fitness=False)	
ga.fitness_function = fitness

ga.run()   

###############################################################################
'''                  Get GA outputs and calculate all cost and 
                     other output featurs      '''
###############################################################################
sol_fitness, sol_df = ga.best_individual()

_, sol_df = npi.group_by(sol_df[:,0]).max(sol_df)
calc_starttime(sol_df)
set_const(sol_df, const)

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
diff_full_time = end_plan - end_time

cost =((coh_fultm*cost_fultm) + 
           (coh_lntm*cost_lntm) + 
           (coh_cnt*cost_cnt) + 
           (coh_dffRqTime*cost_rq_time)+
           (coh_dffVisTime*cost_vis_time)
           )    
#    print(cost)
msk = np.isin(const[:,0], sol_df[:,0])
notUsed_const = const[~msk]
penalty = np.sum(notUsed_const[:,1]) / tot_lenTimeConst  


###############################################################################
'''                  Create query for inser plan in db                      '''
###############################################################################  
tags = 'test'
comment = 'test'
present_id = uuid.uuid1()
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
									   start_time,
									   end_time,
									   dist_len,
									   points_len,
									   duration_len,
									   tags,
									   comment)
                 values ({0}, {1}, 
                         {2}, {3}, {4}, {5},
                         {6}, {7}, {8}, {9}, 
                         {10}, {11},
                         {12}, {13}, {14}, 
                         {15}, {16}) 
               '''.format(city, "'"+str(present_id)+"'",
                          coh_fultm, coh_lntm, coh_cnt, coh_dffRqTime, 
                          cost_fultm, cost_lntm, cost_cnt, cost_rq_time,
                          start_time, end_time,
                          all_dist, len_pln, all_duration,
                          "'"+str(tags)+"'", "'"+str(comment)+"'"
                          )

engine.execute(query_plan)               

inserted_plan = pd.read_sql_query('''SELECT * 
                                  FROM plan_plan
                                  WHERE present_id = {0}
                                  '''.format( "'"+str(present_id)+"'")
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
    

