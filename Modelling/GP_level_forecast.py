# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:36:27 2019
@author: Rajas Khokle
Purpose: To create a csv file that contains t+1 forecast for the drug for 
         differnt practices
"""

'''
------------- Pseudocode ----------------------------------------

- Create a function to load the data from database filtered by drug
- create a function for time series forecasting using fbprophet
- load the drug
- separate the data by practice
- for every practice:
-   create the time series for that practice
-   call the forecasting function
-   Append the t+1 Y_hat and confidence bounds in the dataframe.
- save the final predicted dataframe in the csv file
'''
 # Import the Libraries
 
import pandas as pd
from sqlalchemy import create_engine
from fbprophet import Prophet


# Create Connection to the database

engine = create_engine('postgres://postgres:DataAdmin@127.0.0.1:5432/Capstone')

# Drug loading function.
def load_drug(drug):
    
    sql_string = '''SELECT sum(quantity) as y,period,tranpractice FROM "Casptone_Tableau" WHERE TRANBNFCODE = '''+drug+ 'group by tranpractice, period '
    df = pd.read_sql(sql_string,engine)
    df['ds'] = pd.to_datetime(df.period, format = '%Y%m',errors = 'coerce')
    df.drop('period',axis =1,inplace = True)
    return(df)

# fbprophet forecasting function
def prophetmodel(ts,forecast_period=1):          # Default of t+1 forecast
    model = Prophet()
    model.fit(ts)
    future = model.make_future_dataframe(periods=forecast_period,freq='M') 
    forecast = model.predict(future)
    return(forecast.iloc[-1])

# Define the Drug
lan_pen = "'0601012V0BBAEAD'"
df = load_drug(lan_pen) 
practices = list(df.tranpractice.unique())
demand = []
for i in range(len(practices)):
    ts=df[df.tranpractice==practices[i]]
    try:
        forecast = list(prophetmodel(ts[['ds','y']]) [['ds','yhat','yhat_lower','yhat_upper']])
        demand.append(forecast)
    except:
        continue

saved_df = pd.DataFrame(data = demand, columns= ['Time','Predicted Quantity', 'Lower Bound', 'Higher Boundary'])

saved_df.to_csv('Lantus_Demand_Forecast.csv')



    

