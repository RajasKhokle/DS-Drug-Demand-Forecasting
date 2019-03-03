# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 20:55:31 2019
@author: Rajas Khokle
Purpose: Code Stub to load the extracted Capstone.csv file to Postgresql 
         database. 
"""

import pandas as pd
from sqlalchemy import create_engine

# Create Connection to the database
engine = create_engine('postgres://postgres:DataAdmin@127.0.0.1:5432/Capstone')

df = pd.read_csv('Capstone.csv')
df['period'] =df.period.astype(str)
df.to_sql('Capstone',engine,if_exists='replace',index =True)

