##################################################################
#
# Yuhan Huang, Jan 2020
#
# This script is used for adding some useful information, including distance, speed, acceleration and user ID.
# It searches the csv files in the folder ./Geolife Trajectories 1.3/Data/csv_files, and export the preprocessed csv files into the folder csv_files_preprocessed.
#
# Put this script into the folder ./Geolife Trajectories 1.3, and type "python data_preprocessor.py" in the command line.
#
# There are three options to set:
# 1. The number of csv files in the floder csv_files is not equal to the number of users, type y to continue preprocessing for these files.
# 2. It allows deleting the original files automatically after preprocessing, because I have come up with the problem of memory.
# 3. It allows continuing the work from last interruption. It checks the folder csv_files_preprocessed, and skips the existing files.
#
# Take a cup of coffee and have fun :)
#
##################################################################


import pandas as pd
import geopy.distance
import glob

from datetime import datetime

import re

import sys
import time
import os
from functools import wraps


def logit(func):
	@wraps(func)
	def with_logging(*args, **kwargs):
		print('Start {}...'.format(func.__name__))
		output = func(*args, **kwargs)
		print('\nComplete {}!'.format(func.__name__))
		return output
	return with_logging


def _calculations(df: pd.DataFrame) -> pd.DataFrame:
	distances = []
	speeds = []
	accelerations = []

	for i, row in df.iterrows():
		# calculate distance
		if i == 0: distances.append(0)
		else:
			last_row = df.iloc[i - 1]
			duration = row['datetime'] - last_row['datetime']

			if duration.seconds > 300:		# 5 minutes
				distances.append(0)
			else:
				point1 = (row['latitude'], row['longitude'])
				point2 = (last_row['latitude'], last_row['longitude'])
				try:
					d = geopy.distance.geodesic(point1, point2).m
				except ValueError:
					d = 0
				distances.append(d)

		# calculate speed
		if i == 0: speeds.append(0)
		else:
			last_row = df.iloc[i - 1]
			duration = row['datetime'] - last_row['datetime']
			try:
				speed = distances[-1] / duration.seconds
				speed *= 3.6
			except ZeroDivisionError:
				speed = 0
			speeds.append(speed)

		# calculate acceleration
		if i == 0: accelerations.append(0)
		else:
			last_row = df.iloc[i - 1]
			duration = row['datetime'] - last_row['datetime']
			try:
				acceleration = 0 if distances[-1] == 0 or duration.seconds == 0 else (speeds[-1] - speeds[-2]) / 3.6 / duration.seconds
			except ZeroDivisionError:
				acceleration = 0
			accelerations.append(acceleration)

	assert(len(df) == len(distances) and len(df) == len(speeds) and len(df) == len(accelerations))
	df['distance'] = distances
	df['speed'] = speeds
	df['acceleration'] = accelerations

	return df


def _preprocess_one_user(csv_file, path_new):
	user_id = re.search(r'csv_files/(.*)\.csv', csv_file).group(1)

	with open(csv_file, 'r') as f:
		df = pd.read_csv(f)

	df['datetime'] = list(map(lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'), df['datetime'].values))
	df['user_id'] = user_id

	df = _calculations(df)

	df.to_csv(path_new + '/{}.csv'.format(user_id), index=False)


def _get_param(user_input):
	if user_input == 'y' or user_input == 'Y':
		return True
	elif user_input == 'n' or user_input == 'N':
		return False
	else:
		print('Please input again.')
		return None


@logit
def data_preprocessor(path_csv, path_new, delete=False, conti=False):
	num_user = 0
	for csv_file in glob.glob(path_csv + "/*.csv"):
		num_user += 1

	search = glob.glob(path_new + '/*.csv') if conti else None

	count = 0
	loading = '-\\|/'
	for csv_file in glob.glob(path_csv + "/*.csv"):
		user_id = re.search(r'csv_files/(.*)\.csv', csv_file).group(1)

		if not search or path_new + '/{}.csv'.format(user_id) not in search:
			_preprocess_one_user(csv_file, path_new)

		print('\rSuccess with user ' + user_id)
		if delete:
			try:
				os.remove(path_csv + '/{}.csv'.format(user_id))
				print('\rDelete csv file of user ' + user_id)
			except:
				print('\rError with deleting csv file of user ' + user_id)

		count += 1
		perc = round(count / num_user * 100, 2)
		sys.stdout.write('\r' + ' ' * 50)
		sys.stdout.write('\r{} {}/{} | {}'.format(loading[count % 4], count, num_user, perc))


def csv_detector(path_csv, path_data):
	num_csv = 0
	for csv_file in glob.glob(path_csv + '/*.csv'):
		num_csv += 1

	num_user = 0
	for csv_file in glob.glob(path_data + "/[0-9]*"):
		num_user += 1

	if num_csv != num_user:
		user_input = input('The number of csv files is less than the number of users, do you want to continue? ({}/{}) (y/n)\n'.format(num_csv, num_user))
		if _get_param(user_input) != None:
			return user_input
	else:
		return True


def user_input():
	user_input = input('Do you want to delete the original file after preprocessing? (y/n)\n')
	while True:
		delete = _get_param(user_input)
		if delete != None:
			break

	user_input = input('Do you want to continue from last work? (y/n)\n')
	while True:
		conti = _get_param(user_input)
		if conti != None:
			break

	return delete, conti


def main():
	path_data = "./Data"
	path_csv = path_data + "/csv_files"
	path_new = path_data + "/csv_files_preprocessed"

	# make dir
	folder = os.path.exists(path_new)
	if not folder:
		os.makedirs(path_new)

	if not csv_detector(path_csv, path_data):
		sys.exit(0)

	delete, conti = user_input()

	data_preprocessor(path_csv, path_new, delete=delete, conti=conti)

	sys.exit(0)


if __name__ == "__main__":
	main()
