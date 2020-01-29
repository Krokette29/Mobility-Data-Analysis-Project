import geopandas as gpd
import pandas as pd

import glob

import re
import sys

from datetime import datetime


path = "./Data/csv_files/"

# single csv file import
def csv_importer(csv_file):
    with open(csv_file, 'r') as f:    
        df = pd.read_csv(f)
        df['user_id'] = re.search('\d{3}', csv_file).group()

    return df


# combine all csv files into one dataframe
def multi_csv_importer(path):
    total_num = 0
    for csv_file in glob.glob(path + '*.csv'): total_num += 1
    
    df = pd.DataFrame()
    count = 0
    for csv_file in glob.glob(path + '*.csv'):
        df = df.append(csv_importer(csv_file))
        count += 1
        sys.stdout.write('\r' + '{}/{}'.format(count, total_num))
        
    df = df.sort_values(by=['user_id', 'datetime']) 
    
    return df


print('\nExporting to one csv file...')
df_all = multi_csv_importer(path)
df_all.to_csv(path + 'csv_all.csv', index=False)

print('csv_all.csv has been put in the folder ./Data/csv_files')
