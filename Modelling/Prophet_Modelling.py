# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 19:48:23 2019
@author: Rajas Khokle
Purpose: Demand Forecast Modelling Using FB Prophet
"""

import pandas as pd
from sqlalchemy import create_engine
from fbprophet import Prophet
import matplotlib.pyplot as plt


# Create Connection to the database

engine = create_engine('postgres://postgres:DataAdmin@127.0.0.1:5432/Capstone')

# Drug loading function.
def load_drug(drug):
    
    sql_string = '''SELECT sum(quantity),period FROM "Casptone_Tableau" WHERE TRANBNFCODE = '''+drug+ 'group by period '
    df = pd.read_sql(sql_string,engine)
    df['dt'] = pd.to_datetime(df.period, format = '%Y%m',errors = 'coerce')
    ds=df['dt']                  # Column for datestamp in Prophet model
    y = df['sum']                # Column for timeseries in prohet model
    data_dict = {'ds':ds,'y':y}
    ts = pd.DataFrame(data_dict)
    ts.reset_index(inplace =True,drop =True)
    return(ts)
    
# FB Prophet Modelling function

def prophetmodel(ts,forecast_period=12):
    # Train Test Split 
    train = ts[:-12]   # leave out last twelve points for testing 
    model = Prophet()
    model.fit(train)
    future = model.make_future_dataframe(periods=forecast_period,freq='M') 
    forecast = model.predict(future)
    model.plot(forecast)
    plt.plot(ts['ds'],ts['y'])
    plt.plot(forecast['ds'],forecast['yhat'])
    plt.show()
    return(model,forecast)

months = 24

# Model for Insulin Humulin M3_KwikPen 
hum_pen = "'0601012D0BBAVBZ'"
ts = load_drug(hum_pen) 
model,forecast = prophetmodel(ts,months)


# Model Insulin Lantus SoloStar_100u/ml 3ml Pf Pen
lan_pen = "'0601012V0BBAEAD'"
ts = load_drug(lan_pen) 
model,forecast = prophetmodel(ts,months)


# Model Insulin Humulin M3_100u/ml 3ml Cartridge 
hum_cart = "'0601012V0AAAAAA'"
ts = load_drug(hum_cart) 
model,forecast = prophetmodel(ts,months)


# Model Insulin Lantus_100u/ml 3ml Cartridge 
lant_cart = "'0601012D0BBASBA'"
ts = load_drug(lant_cart) 
model,forecast = prophetmodel(ts,months)


# Model Metformin
metformin = "'0601022B0AAABAB'"
ts = load_drug(metformin) 
model,forecast = prophetmodel(ts,months)


# Model Orlistat 120 mg
orlistat_120 = "'0405010P0AAAAAA'"
ts = load_drug(orlistat_120) 
model,forecast = prophetmodel(ts,months)

# Model Orlistat 120 mg without anomaly
orlistat_120 = "'0405010P0AAAAAA'"
ts = load_drug(orlistat_120) 
model,forecast = prophetmodel(ts[26:],months)


# Model Orlistat 60 mg
orlistat_60 = "'0405010P0AAABAB'"
ts = load_drug(orlistat_60) 
model,forecast = prophetmodel(ts,months)

# Model Orlistat 60 mg Without Anomaly
orlistat_60 = "'0405010P0AAABAB'"
ts = load_drug(orlistat_60) 
model,forecast = prophetmodel(ts[26:],months)
