import pandas as pd
import glob

from datetime import datetime

import re

import sys
import time
import os


# import one single plt file
def _plt_importer(file_name: str) -> pd.DataFrame:
    file = open(file_name, 'r')
    df_raw = pd.read_table(file)
    
    date_time = []
    latitude = []
    longitude = []
    altitude = []
    days_elapsed = []
    
    for i, row in df_raw.iterrows():
        data_list = row['Geolife trajectory'].split(',')
        if len(data_list) != 7: continue
    
        latitude.append(float(data_list[0]))
        longitude.append(float(data_list[1]))
        altitude.append(float(data_list[3]))
        days_elapsed.append(data_list[4])
        date = data_list[5]
        time = data_list[6]
        date_time.append(datetime.strptime(date+' '+time, '%Y-%m-%d %H:%M:%S'))
        
    df = pd.DataFrame(data={'datetime':date_time, 'altitude':altitude, 'days_elapsed':days_elapsed, 'latitude':latitude, 'longitude':longitude})
    
    df = df.sort_values(by='datetime')
    
    return df


# import plt files of one user, and export as a user_id.csv file
def _user_plt_transformer(user_folder: str, folder: str):
	user_id = re.search(r'Data/(.*)', user_folder).group(1)
	df_all = pd.DataFrame()
	for plt_file in glob.glob(user_folder + '/Trajectory/*.plt'):
		df_single = _plt_importer(plt_file)
		df_all = df_all.append(df_single)

	df_all = df_all.sort_values(by='datetime')
	df_all.to_csv(folder + '/csv_files/{}.csv'.format(user_id), index=False)
    

# transform all plt files to multiple user_id.csv file
def data_transformer(folder: str, conti=False):
    # check the total number of users
    num_user = 0
    for user_folder in glob.glob(folder + '/[0-9]*'):
        num_user += 1
        
    print('Start transforming...')

    search = glob.glob(folder + '/csv_files/*.csv') if conti else None
        
    count = 0
    loading = '-\\|/'
    for user_folder in glob.glob(folder + '/[0-9]*'):
        user_id = re.search(r'Data/(.*)', user_folder).group(1)
        if not search or folder + '/csv_files/' + user_id + '.csv' not in search:
                _user_plt_transformer(user_folder, folder)

        count += 1
        perc = round(count / num_user * 100, 2)
        sys.stdout.write('\r' + ' ' * 50)
        sys.stdout.write('\r{} {}/{} | {}%'.format(loading[count % 4], count, num_user, perc))
    
    print('\nTransform complete!')

while True:
	# make dir
	path = './Data'
	csv_path = path + '/csv_files'
	folder = os.path.exists(csv_path)
	if not folder:
		os.makedirs(csv_path)

	user_input = input('Do you want to continue from last work? (y/n)\n')
	if user_input == 'y' or user_input == 'Y':
		conti = True
	elif user_input == 'n' or user_input == 'N':
		conti = False
	else:
		print('Please input again.')
		continue

	data_transformer(path, conti=conti)
	break

sys.exit(0)
