# -*- coding: utf-8 -*-
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import requests

city = 36 #  'استانبول'

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

df = pd.read_sql_query('''SELECT 
                        	id,latt, "long", city_id
                        FROM 
                        	public.plan_attractions
                        WHERE
                        	city_id = {}
                       '''.format(city)
                       ,con=engine)


url = 'https://api.neshan.org/v2/direction?'
apiKey = 'service.rstJXLArDfrfB3GG2iLd3i08trxmzNP1gjKd4lEI'
headers = {"Accept": "application/json", "Api-Key":apiKey}

l = len(df)
for i in range(l):        
    for j in range(l):        
        url = 'https://api.neshan.org/v2/direction?'
        destination_id = df.loc[j,'id']
        qry = '''insert into plan_distance_mat(ecl_dist, len_meter, len_time, 
                 destination_id, origin_id, travel_type_id)
                 values({},{},{},{},{},{})
              '''
        origin = str(df.loc[i,'latt']) + ',' + str(df.loc[i,'long'])
        ecl_dist, len_meter, len_time, travel_type_id = 0,0,0,1
        origin_id = df.loc[i,'id']
        
        if i!=j:
            destin = str(df.loc[j,'latt']) + ',' + str(df.loc[j,'long'])
            url = url + 'origin=' + origin + '&destination=' + destin
            r = requests.get(url, headers = headers)
            data = r.json() 
            if data['routes']:
                route = data['routes'][0]['legs'][0]
                len_time = route['duration']['value']//60
                len_meter = route['distance']['value']
                ecl_dist = np.sqrt((df.loc[i,'latt']-df.loc[j,'latt'])**2 +
                            (df.loc[i,'long']-df.loc[j,'long'])**2)
            
        qry = qry.format(ecl_dist, len_meter, len_time,
                         destination_id, origin_id, travel_type_id)
        print('i= ' + str(i) + ' - j= ' + str(j))
        engine.execute(qry) 

