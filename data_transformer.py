####################################################################################################
#
# Yuhan Huang, Jan 2020
#
# This script is used for collecting all plt files from every user, including detecting the label files and add labels, if they exist.
# Then export a csv file for every user,  and store all csv files in a seperate folder, i.e. ./Data/csv_files
#
# Put this script into the folder ./Geolife Trajectories 1.3, and type "python data_transformer.py" in the command line.
# Make sure you have already cd to the same folder as the script.
# If the action is denied because of no permission, try "chmod 777 data_transformer.py".
#
# Because it takes quite a long time to transform so many files, so I add a progress bar.
# The script will ask you "Do you want to continue from last work? (y/n)".
# If this is the first time excuting the script, just type in 'n'.
# If the work is interrupted because of some certain reasons, you can type in 'y' when next time executing this script.
#
# Take a cup of coffee and have fun :)
#
####################################################################################################

import pandas as pd
import glob

from datetime import datetime

import re

import sys
import time
import os


# import one single plt file
def _plt_importer(file_name: str) -> pd.DataFrame:
    with open(file_name, 'r') as f:
    	df_raw = pd.read_table(f)
    
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

	# add labels
	df_all['mode'] = None
	label_exist = False
	try:
		with open(user_folder + '/labels.txt') as f:
			lines = f.readlines()
		label_exist = True
	except:
		pass

	if label_exist:
		try:
			for i in range(1, len(lines)):
				line = lines[i]
				line_list = line.split('\t')
				line_list[2] = line_list[2][:-1]

				start = line_list[0]
				end = line_list[1]
				mode = line_list[2]
				df_all.loc[(start <= df_all['datetime']) & (df_all['datetime'] <= end), 'mode'] = mode

			print('\rAdd labels for user ' + user_id)
		except:
			print('\rError with labels for user ' + user_id)


	# export to csv file
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


def main():
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

if __name__ == '__main__':
	main()
