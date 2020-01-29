##################################################################
#
# Yuhan Huang, Jan 2020
#
# This script is used for exporting one csv file including the information of average speed/acceleration and maximum speed/acceleration for every track.
#
##################################################################

import pandas as pd
from datetime import datetime
import sys
import time
import numpy as np


def data_importer(path, all=False, label=True):
	csv_file_path = path

	with open(csv_file_path, 'r') as f:
		print('Importing data...')
		df = pd.read_csv(f)

		if not all and label:
			df = df.loc[list(map(lambda x: isinstance(x, str), df['mode']))]
		elif not all and not label:
			df = df.loc[list(map(lambda x: not isinstance(x, str), df['mode']))]
	
	# gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

	print('Import complete! Wait a little moment for some other settings...')

	# change user id from float to str
	df['user_id'] = list(map(lambda x: '0' * (3 - len(str(int(x)))) + str(int(x)), df['user_id']))

	# change datetime from str to datetime.datetime
	df['datetime'] = list(map(lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'), df['datetime'].values))

	# sort all values according to user ID and datetime
	df = df.sort_values(by=['user_id', 'datetime'])
	df.index = [i for i in range(len(df))]

	print('Import all complete!')
	return df


def calculate_box_plot_characteristics(my_list: list):
    result = {}
    
    result["minimum"] = np.min(my_list)
    result["maximum"] = np.max(my_list)
    result["median"] = np.median(my_list)
    
    q1 = np.percentile(my_list, 25)
    q3 = np.percentile(my_list, 75)
    iqr = q3 - q1
    result["lower_quartile"] = q1
    result["upper_quartile"] = q3
    
    lower_whisker = q1 - 1.5 * iqr
    upper_whisker = q3 + 1.5 * iqr
    rpa_sort = np.sort(my_list)
    for i in range(len(rpa_sort)):
        if rpa_sort[i] > lower_whisker:
            result["lower_whisker"] = rpa_sort[i]
            break
        if i == len(rpa_sort) - 1:
        	result["lower_whisker"] = lower_whisker
        	break
    for i in reversed(range(len(rpa_sort))):
        if rpa_sort[i] < upper_whisker:
            result["upper_whisker"] = rpa_sort[i]
            break
        if i == 0:
        	result["upper_whisker"] = upper_whisker
        	break
    
    return result


def metadata_exporter(df):
	speeds = []
	accs = []

	speed_avg = []
	acc_avg = []
	speed_max = []
	acc_max = []
	label = []
	user_id = []
	end_time = []


	for i, row in df.iterrows():
		assert(len(speed_avg) == len(acc_avg) == len(speed_max) == len(acc_max) == len(label) == len(user_id) == len(end_time))		
		if i % 100 == 0: sys.stdout.write('\r{} / {}'.format(i, len(df)))
		if i == 0 or i == len(df) - 1: 
			speeds.append(row['speed'])
			accs.append(row['acceleration'])

		else:
			last_row = df.iloc[i - 1]
			duration = row['datetime'] - last_row['datetime']
			if duration.seconds > 300 or row['mode'] != last_row['mode'] or row['user_id'] != last_row['user_id']:
				if speeds and accs:
					speed_avg.append(np.mean(speeds))
					acc_avg.append(np.mean(accs))
					speed_max.append(calculate_box_plot_characteristics(speeds)['upper_whisker'])
					acc_max.append(calculate_box_plot_characteristics(accs)['upper_whisker'])
					label.append(last_row['mode'])
					user_id.append(last_row['user_id'])
					end_time.append(last_row['datetime'])

				speeds = []
				accs = []

			speeds.append(row['speed'])
			accs.append(row['acceleration'])

	if speeds and accs:
		speed_avg.append(np.mean(speeds))
		acc_avg.append(np.mean(accs))
		speed_max.append(calculate_box_plot_characteristics(speeds)['upper_whisker'])
		acc_max.append(calculate_box_plot_characteristics(accs)['upper_whisker'])
		label.append(df.iloc[len(df)-1]['mode'])
		user_id.append(df.iloc[len(df)-1]['user_id'])
		end_time.append(df.iloc[len(df)-1]['datetime'])

	assert(len(speed_avg) == len(acc_avg) == len(speed_max) == len(acc_max) == len(label) == len(user_id) == len(end_time))	
	metadata_df = pd.DataFrame(data={'average_speed':speed_avg, 'average_acceleration':acc_avg, 'max_speed':speed_max, 'max_acceleration':acc_max, 'mode':label, 'user_id':user_id, 'end_time':end_time})

	print('\nCalculate compelete!')

	return metadata_df


def main():
	path = './df_all.csv'
	df = data_importer(path, all=False, label=True)
	metadata_df = metadata_exporter(df)

	print('Writing into metadata_df.csv...')
	metadata_df.to_csv('./metadata_df.csv', index=False)
	print('All complete!')

	sys.exit(0)


if __name__ == '__main__':
	main()