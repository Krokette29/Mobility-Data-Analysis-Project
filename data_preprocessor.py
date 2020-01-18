import pandas as pd
import geopy.distance
import glob

from datetime import datetime

import re

import sys
import time
import os
from functools import wraps


def calculate_distance(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Metric: m
	"""
	distances = []
	for i, row in df.iterrows():
		if i == 0: distances.append(0)
		else:
			last_row = df.iloc[i - 1]
			duration = row['datetime'] - last_row['datetime']

			if duration.seconds > 300:		# 5 minutes
				distances.append(0)
			else:
				point1 = (row['latitude'], row['longitude'])
				point2 = (last_row['latitude'], last_row['longitude'])

				d = geopy.distance.great_circle(point1, point2).m
				distances.append(d)

	assert(len(df) == len(distances))
	df['distance'] = distances
	return df


def calculate_speed(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Metric: km/h
	"""
	speeds = []
	for i, row in df.iterrows():
		if i == 0: speeds.append(0)
		else:
			last_row = df.iloc[i - 1]
			duration = row['datetime'] - last_row['datetime']
			try:
				speed = row['distance'] / duration.seconds
				speed *= 3.6
			except ZeroDivisionError:
				speed = 0
			speeds.append(speed)

	assert(len(df) == len(speeds))
	df['speed'] = speeds
	return df
	

def calculate_acceleration(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Metric: m/s^2
	"""
	accelerations = []
	for i, row in df.iterrows():
		if i == 0: accelerations.append(0)
		else:
			last_row = df.iloc[i - 1]
			duration = row['datetime'] - last_row['datetime']
			try:
				acceleration = 0 if row['distance'] == 0 else (row['speed'] - last_row['speed']) / 3.6 / duration.seconds
			except ZeroDivisionError:
				acceleration = 0
			accelerations.append(acceleration)

	assert(len(df) == len(accelerations))
	df['acceleration'] = accelerations
	return df


def main():
	print('Start preprocessing...')

	path_data = "./Data"
	path_csv = path_data + "/csv_files"
	path_new = path_data + "/csv_files_preprocessed"

	# make dir
	folder = os.path.exists(path_new)
	if not folder:
		os.makedirs(path_new)

	num_user = 0
	for csv_file in glob.glob(path_csv + "/*.csv"):
		num_user += 1

	count = 0
	loading = '-\\|/'
	for csv_file in glob.glob(path_csv + "/*.csv"):
		user_id = re.search(r'csv_files/(.*)\.csv', csv_file).group(1)
		with open(csv_file, 'r') as f:
			df = pd.read_csv(f)

		df['datetime'] = list(map(lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'), df['datetime'].values))
		df['user_id'] = user_id

		df = calculate_distance(df)
		df = calculate_speed(df)
		df = calculate_acceleration(df)

		df.to_csv(path_new + '/{}.csv'.format(user_id), index=False)

		count += 1
		perc = round(count / num_user * 100, 2)
		sys.stdout.write('\r' + ' ' * 50)
		sys.stdout.write('\r{} {}/{} | {}%'.format(loading[count % 4], count, num_user, perc))

	print('Preprocessing complete!')




if __name__ == "__main__":
	main()
