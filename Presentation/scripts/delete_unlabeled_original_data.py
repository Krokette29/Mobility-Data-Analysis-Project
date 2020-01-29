import glob
import re
import sys
import os
import shutil


def delete_unlabeled_original_data(delete=False):
	path_data = "./Data/"
	path_csv = path_data + "csv_files/"
	path_new = path_data + "csv_files_preprocessed/"

	for user_path in glob.glob(path_data + "[0-9]*/"):
		if not os.path.exists(user_path + "labels.txt"):
			if os.path.exists(user_path + "Trajectory/"):
				if delete:
					shutil.rmtree(user_path + "Trajectory/")
				print('Delete data from user ' + user_path[-4:-1])
			else:
				print('No original data for user ' + user_path[-4:-1])
		else:
			print('Labeled data cannot be deleted from user ' + user_path[-4:-1])


def main():
	user_input = input('First run the checking process. It will only shows which user files will be deleted. The delete actions will NOT be activated now. Press enter to continue...\n')
	delete_unlabeled_original_data(delete=False)

	print('--------------------------')
	user_input = input('Checking process is done. Check for the information above, and type in DELETE to activate the delete actions. CAUTION! Thess actions cannot be reversed!\n')

	if user_input == 'DELETE':
		delete_unlabeled_original_data(delete=True)
		print('--------------------------')
		print('Deletions complete. Exit.')
	else:
		print('No deletions. Exit.')
	sys.exit(0)


if __name__ == '__main__':
	main()
