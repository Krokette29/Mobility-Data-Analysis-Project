import glob
import shutil



for file in glob.glob("./Data/[0-9]*/"):
	# print(file)
	shutil.rmtree(file)