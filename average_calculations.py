import pandas as pd
from datetime import datetime
import sys
import time


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


def average_calculations(df):
	speed_avg = []
	acc_avg = []
	speed_sum = 0
	acc_sum = 0
	counter = 0

	for i, row in df.iterrows():
		if i % 100 == 0: sys.stdout.write('\r{} / {}'.format(i, len(df)))
		if i == 0 or i == len(df) - 1: 
			speed_sum += row['speed']
			acc_sum += row['acceleration']
			counter += 1

		else:
			last_row = df.iloc[i - 1]
			duration = row['datetime'] - last_row['datetime']
			if duration.seconds > 300 or row['mode'] != last_row['mode'] or row['user_id'] != last_row['user_id']:
				speed_avg += [speed_sum / counter] * counter
				acc_avg += [acc_sum / counter] * counter
				speed_sum = 0
				acc_sum = 0
				counter = 0
			speed_sum += row['speed']
			acc_sum += row['acceleration']
			counter += 1

	speed_avg += [speed_sum / counter] * counter
	acc_avg += [acc_sum / counter] * counter

	assert(len(df) == len(speed_avg) and len(df) == len(acc_avg))
	df['average_speed_current_track'] = speed_avg
	df['average_acceleration_current_track'] = acc_avg
	print('\nCalculate compelete!')

	return df


def main():
	path = './df_all.csv'
	df = data_importer(path, all=False, label=True)
	df = average_calculations(df)

	print('Writing into df_all_2.csv...')
	df.to_csv('./df_all_2.csv', index=False)
	print('All complete!')

	sys.exit(0)


if __name__ == '__main__':
	main()
	