# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:59:24 2019
@author: Rajas Khokle
Purpose: Get the data for insulin pumps, cartridges, metformin and orlistat 
from all the csv files and load it into a database for consumption by Tableau. 
"""

import pandas as pd
from sqlalchemy import create_engine


# Create Connection to the database

engine = create_engine('postgres://postgres:DataAdmin@127.0.0.1:5432/Capstone')


# Create the filenames of the CSV files that contain NHS GP prescription dsata
year = list(range(2010,2019))
year = [str(x) for x in year]
month= list(range(1,13))
month = ['0'+str(x) if x in range(1,10) else str(x) for x in month ]


pdpi_fullfile=[]
for i in range(len(year)):
    for j in range(len(month)):
        pdpi_fullfile.append('../Capstonedata/T'+year[i]+month[j]+'PDPI BNFT'+'.csv')

trans_columns = ['tranpractice','tranbnfcode','tranbnfname','items','nic',
                    'actcost','quantity','period']
# Extract only those entries that correspond to Metformin, Insulin Pump and Orlistat.
# Load them into Capstone_Tableu table in Postgresql database.
for i in range(len(pdpi_fullfile)):
    try:
        df_trans = pd.read_csv(pdpi_fullfile[i])
    except:
        continue
    df_trans.drop([df_trans.columns[0],df_trans.columns[1],df_trans.columns[-1]], 
                  axis = 1,inplace =True)
    df_trans.columns = trans_columns
    print(f'Processing {i} th file')
    
    hum_cart = df_trans[df_trans['tranbnfcode'].str.match('0601012D0BBASBA')]
    hum_pen = df_trans[df_trans['tranbnfcode'].str.match('0601012D0BBAVBZ')]
    lantus_pen = df_trans[df_trans['tranbnfcode'].str.match('0601012V0BBAEAD')]
    gen_cart = df_trans[df_trans['tranbnfcode'].str.match('0601012V0AAAAAA')]
    metformin = df_trans[df_trans['tranbnfcode'].str.match('0601022B0AAABAB')]
    Orli_120 = df_trans[df_trans['tranbnfcode'].str.match('0405010P0AAAAAA')]
    Orli_60 = df_trans[df_trans['tranbnfcode'].str.match('0405010P0AAABAB')]
    
    frames = [hum_cart,hum_pen,lantus_pen,gen_cart,metformin,Orli_60,Orli_120]
    df_total = pd.concat(frames)
    
    df_total.to_sql('Capstone_Tableau',engine,if_exists='append',index =False)
    

    
    
    