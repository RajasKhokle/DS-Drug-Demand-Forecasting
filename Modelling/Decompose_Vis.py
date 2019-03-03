# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:09:34 2019
@author: Rajas Khokle
Purpose: To visualize the seasonal decomposition of the drug sale. 

"""

# Import the Libraries

import pandas as pd
from sqlalchemy import create_engine

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


# Create Connection to the database
engine = create_engine('postgres://postgres:DataAdmin@127.0.0.1:5432/Capstone')


# Drug loading function.
def load_drug(drug):
    
    sql_string = '''SELECT sum(quantity) as quantity,period FROM "Casptone_Tableau" WHERE TRANBNFCODE = '''+drug+ 'group by period '
    df = pd.read_sql(sql_string,engine)
    df['dt'] = pd.to_datetime(df.period, format = '%Y%m',errors = 'coerce')
    QD = df['quantity']
    QD.index = df['dt']
    return(QD)


# Decompose and plot function
def ssdecompose(ts):
    ss_decomposition = seasonal_decompose(ts)
    fig = plt.figure()
    fig.set_size_inches(12,10)
    fig = ss_decomposition.plot()
    plt.show

# Load Insulin Humulin M3_KwikPen 
hum_pen = "'0601012D0BBAVBZ'"
QD = load_drug(hum_pen) 
ssdecompose(QD)

# Load Insulin Lantus SoloStar_100u/ml 3ml Pf Pen
lan_pen = "'0601012V0BBAEAD'"
QD = load_drug(lan_pen) 
ssdecompose(QD)

# Load Insulin Humulin M3_100u/ml 3ml Cartridge 
hum_cart = "'0601012V0AAAAAA'"
QD = load_drug(hum_cart) 
ssdecompose(QD)

# Load Insulin Lantus_100u/ml 3ml Cartridge 
lant_cart = "'0601012D0BBASBA'"
QD = load_drug(lant_cart) 
ssdecompose(QD)

# Load Metformin
metformin = "'0601022B0AAABAB'"
QD = load_drug(metformin) 
ssdecompose(QD)

# Load Orlistat 120 mg
orlistat_120 = "'0405010P0AAAAAA'"
QD = load_drug(orlistat_120) 
ssdecompose(QD)

# Load Orlistat 60 mg
orlistat_60 = "'0405010P0AAABAB'"
QD = load_drug(orlistat_120) 
ssdecompose(QD)
