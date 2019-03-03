# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 17:50:57 2019
@author: Rajas Khokle
Purpose: To collect data from all the NHS GP prescription CSV files, split them into 
         BNF categories and save them in differnt tables in a database for faster 
		 access and searching through the database.
"""

# Import the libraries

import pandas as pd
from sqlalchemy import create_engine


# Create Connection to the database

engine = create_engine('postgres://postgres:DataAdmin@127.0.0.1:5432/Capstone')


# Form the filename
year = list(range(2010,2019))
year = [str(x) for x in year]

month= list(range(1,13))
month = ['0'+str(x) if x in range(1,10) else str(x) for x in month ]


pdpi_fullfile=[]
for i in range(len(year)):
    for j in range(len(month)):
        pdpi_fullfile.append('../Capstonedata/T'+year[i]+month[j]+'PDPI BNFT'+'.csv')



# Load the Transaction File, separate by categories and store in dB.
trans_columns = ['tranpractice','tranbnfcode','tranbnfname','items','nic',
                    'actcost','quantity','period']
for i in range(len(pdpi_fullfile)):
    try:
        df_trans = pd.read_csv(pdpi_fullfile[i])
    except:
        continue
    df_trans.drop([df_trans.columns[0],df_trans.columns[1],df_trans.columns[-1]], 
                  axis = 1,inplace =True)
    df_trans.columns = trans_columns
    print(f'Processing {i} th file')
    
    # Separate into categories according to BNF Codes
    
    cat1 = df_trans[df_trans['tranbnfcode'].str.match('01')]
    cat2 = df_trans[df_trans['tranbnfcode'].str.match('02')]
    cat3 = df_trans[df_trans['tranbnfcode'].str.match('03')]
    cat4 = df_trans[df_trans['tranbnfcode'].str.match('04')]
    cat5 = df_trans[df_trans['tranbnfcode'].str.match('05')]
    cat6 = df_trans[df_trans['tranbnfcode'].str.match('06')]
    cat7 = df_trans[df_trans['tranbnfcode'].str.match('07')]
    cat8 = df_trans[df_trans['tranbnfcode'].str.match('08')]
    cat9 = df_trans[df_trans['tranbnfcode'].str.match('09')]
    cat10 = df_trans[df_trans['tranbnfcode'].str.match('10')]
    cat11 = df_trans[df_trans['tranbnfcode'].str.match('11')]
    cat12 = df_trans[df_trans['tranbnfcode'].str.match('12')]
    cat13 = df_trans[df_trans['tranbnfcode'].str.match('13')]
    cat14 = df_trans[df_trans['tranbnfcode'].str.match('14')]
    cat15 = df_trans[df_trans['tranbnfcode'].str.match('15')]
    cat18 = df_trans[df_trans['tranbnfcode'].str.match('18')]
    cat19 = df_trans[df_trans['tranbnfcode'].str.match('19')]
    cat20 = df_trans[df_trans['tranbnfcode'].str.match('20')]
    cat21 = df_trans[df_trans['tranbnfcode'].str.match('21')]
    cat22 = df_trans[df_trans['tranbnfcode'].str.match('22')]
    cat23 = df_trans[df_trans['tranbnfcode'].str.match('23')]
    
    # Save to Tables
    
    cat1.to_sql('cat1',engine,if_exists = 'append',index=True)
    cat2.to_sql('cat2',engine,if_exists = 'append',index=True)
    cat3.to_sql('cat3',engine,if_exists = 'append',index=True)
    cat4.to_sql('cat4',engine,if_exists = 'append',index=True)
    cat5.to_sql('cat5',engine,if_exists = 'append',index=True)
    cat6.to_sql('cat6',engine,if_exists = 'append',index=True)
    cat7.to_sql('cat7',engine,if_exists = 'append',index=True)
    cat8.to_sql('cat8',engine,if_exists = 'append',index=True)
    cat9.to_sql('cat9',engine,if_exists = 'append',index=True)
    cat10.to_sql('cat10',engine,if_exists = 'append',index=True)
    cat11.to_sql('cat11',engine,if_exists = 'append',index=True)
    cat12.to_sql('cat12',engine,if_exists = 'append',index=True)
    cat13.to_sql('cat13',engine,if_exists = 'append',index=True)
    cat14.to_sql('cat14',engine,if_exists = 'append',index=True)
    cat15.to_sql('cat15',engine,if_exists = 'append',index=True)
    cat18.to_sql('cat18',engine,if_exists = 'append',index=True)
    cat19.to_sql('cat19',engine,if_exists = 'append',index=True)
    cat20.to_sql('cat20',engine,if_exists = 'append',index=True)
    cat21.to_sql('cat21',engine,if_exists = 'append',index=True)
    cat22.to_sql('cat22',engine,if_exists = 'append',index=True)
    cat23.to_sql('cat23',engine,if_exists = 'append',index=True)
      
  
    




