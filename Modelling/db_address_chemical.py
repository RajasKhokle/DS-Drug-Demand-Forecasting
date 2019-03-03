# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 19:16:53 2019
@author: Rajas Khokle
Purpose: Load the Address of the GP practice and chemical composition into the database. 
"""

# Import the libraries

import pandas as pd
from sqlalchemy import create_engine


# Create Connection to the database

engine = create_engine('postgres://postgres:DataAdmin@127.0.0.1:5432/Capstone')


# Create the Tables - Chemicals and Address.


chem_create = '''CREATE TABLE CHEMICALS(
        CHEMID SERIAL PRIMARY KEY,
        CHEMBNFCODE VARCHAR (15) NOT NULL,
        CHEMBNFNAME VARCHAR(200) NOT NULL
        );'''

addr_create='''CREATE TABLE ADDRESS(
        ADDID SERIAL PRIMARY KEY,
        ADDRDATE VARCHAR(10),
        ADDPRACTICE VARCHAR(10) NOT NULL,
        PRACNAME VARCHAR(200) NOT NULL,
        PRACADDR VARCHAR(200),
        STREETADDR VARCHAR(200),
        AREA VARCHAR(200),
        TOWN VARCHAR(200),
        ZIPCODE VARCHAR (10),
        COUNTRY VARCHAR(20)
        );'''


try:
    engine.execute(chem_create)
except:
    print('The Chemical Table already exists. Deleting existing one to create a new table.')
    engine.execute('DROP TABLE CHEMICALS')
    engine.execute(chem_create)
    

try:
    engine.execute(addr_create)
except:
    print('The Address Table already exists. Deleting existing one to create a new table.')
    engine.execute('DROP TABLE ADDRESS')
    engine.execute(addr_create)
        
    

# Get the filename from the NHS GP Prescribing CSv data.
year = list(range(2010,2019))
year = [str(x) for x in year]
month= list(range(1,13))
month = ['0'+str(x) if x in range(1,10) else str(x) for x in month ]

chem_fullfile=[]
for i in range(len(year)):
    for j in range(len(month)):
        chem_fullfile.append('../Capstonedata/T'+year[i]+month[j]+'CHEM SUBS'+'.csv')        

addr_fullfile=[]
for i in range(len(year)):
    for j in range(len(month)):
        addr_fullfile.append('../Capstonedata/T'+year[i]+month[j]+'ADDR BNFT'+'.csv')   

# Load the chemical File

for i in range(len(chem_fullfile)):
    try:
        df_chem = pd.read_csv(chem_fullfile[i])
    except:
        continue
    
    df_chem.drop([df_chem.columns[2],df_chem.columns[3]],axis = 1,inplace =True) # Drop Unnecessary columns
    df_chem.columns=['chembnfcode','chembnfname']                                # Rename the columns
    
    # load to the database
    df_chem.to_sql('chemicals',engine,if_exists='append',index =False)


    # Remove the duplicates from the chemicals columns

del_dup = '''DELETE  FROM
    chemicals a
        USING chemicals b
WHERE
    a.chemid > b.chemid
    AND a.chembnfname = b.chembnfname;'''

engine.execute(del_dup)

# Load the Address File
addr_columns = ['addrdate','addpractice','pracname','pracaddr','streetaddr','area',
                'town','zipcode']
for i in range(len(chem_fullfile)):
    try:
        df_addr = pd.read_csv(addr_fullfile[i],header = None)
    except:
        continue
    if len(df_addr.columns)>8:                                # To handle some files for which extra null column at end is imported
        df_addr.drop(df_addr.columns[-1],axis =1,inplace =True)
    
    df_addr.columns = addr_columns
    df_addr['country'] = 'United Kingdom'
    # load to the database
    df_addr.to_sql('address',engine,if_exists='append',index =False)


