import geopandas as gpd
import pandas as pd
from datetime import datetime
import glob


def data_importer(all=False, label=True, multi_index=False):
	path = './Data/csv_files_preprocessed/'

	# retrieve all data, with or without label
	gdf_all = gpd.GeoDataFrame()
	for file in glob.glob(path + '[0-9]*.csv'):
		with open(file) as f:
			try:
				df = pd.read_csv(f)
			except:
				print('Import error!')

			if not all and label:
				df = df.loc[list(map(lambda x: isinstance(x, str), df['mode']))]
			elif not all and not label:
				df = df.loc[list(map(lambda x: not isinstance(x, str), df['mode']))]

		if len(df) != 0:
			gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
			gdf_all = gdf_all.append(gdf)
			print('Import data from user {}, length of total label: {}'.format(file[-7:-4], len(gdf_all)))

	print('Import complete! Wait a little moment for some other settings...')

	# change user id from float to str
	gdf_all['user_id'] = list(map(lambda x: '0' * (3 - len(str(int(x)))) + str(int(x)), gdf_all['user_id']))

	# change datetime from str to datetime.datetime
	gdf_all['datetime'] = list(map(lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'), gdf_all['datetime'].values))

	# sort all values according to user ID and datetime
	gdf_all = gdf_all.sort_values(by=['user_id', 'datetime'])
	gdf_all.index = [i for i in range(len(gdf_all))]

	# multi index
	if multi_index:
		gdf_all.index = pd.MultiIndex.from_arrays([gdf_all['user_id'], gdf_all['datetime']], names=['user_id', 'datetime'])

	print('All complete!')

	return gdf_all
