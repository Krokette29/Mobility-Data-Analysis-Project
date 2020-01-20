import geopandas as gpd
import pandas as pd
from datetime import datetime
import glob


def data_importer(all=False, label=True, multi_index=False) -> gpd.GeoDataFrame:
	csv_file_path = './Data/df_all.csv'

	with open(csv_file_path, 'r') as f:
		print('Importing data...')
		df = pd.read_csv(f)

		if not all and label:
			df = df.loc[list(map(lambda x: isinstance(x, str), df['mode']))]
		elif not all and not label:
			df = df.loc[list(map(lambda x: not isinstance(x, str), df['mode']))]
	
	gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

	print('Import complete! Wait a little moment for some other settings...')

	# change user id from float to str
	gdf['user_id'] = list(map(lambda x: '0' * (3 - len(str(int(x)))) + str(int(x)), gdf['user_id']))

	# change datetime from str to datetime.datetime
	gdf['datetime'] = list(map(lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'), gdf['datetime'].values))

	# sort all values according to user ID and datetime
	gdf = gdf.sort_values(by=['user_id', 'datetime'])
	gdf.index = [i for i in range(len(gdf))]

	# multi index
	if multi_index:
		gdf.index = pd.MultiIndex.from_arrays([gdf['user_id'], gdf['datetime']], names=['user_id', 'datetime'])

	print('All complete!')
	return gdf


def convert_to_one_hot(Y, num_features):
    Y = np.eye(num_features)[Y.reshape(-1)].T
    return Y


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
    for i in reversed(range(len(rpa_sort))):
        if rpa_sort[i] < upper_whisker:
            result["upper_whisker"] = rpa_sort[i]
            break
    
    return result
