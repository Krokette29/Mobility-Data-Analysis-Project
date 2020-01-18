# Mobility-Data-Analysis-Project
project for "Mobility Data Analysis"

## 关于data_fransformer.py的使用
该脚本整合每个用户的所有数据为一个pandas dataframe，然后导出成`user_id.csv`文件存放在`Data/csv_files`文件夹下，方便后续读取和处理数据。

把该文件放到目录`mda/07_machine_learning_classification_project/project/Geolife Trajectories 1.3/`下，即和 Data 文件夹放在一起。然后运行服务器终端，输入以下命令即可开始运行：  
`python /mda/07_machine_learning_classification_project/project/Geolife\ Trajectories\ 1.3/data_transformer.py`

一开始询问是否从上一次工作继续，输入n表示重新开始（第一次运行），输入y表示上一次转换工作被中断，从上一次接着继续。开始运行后，会出现转换进度条显示当前转换进度。

若出现permission denied的情况，命令行中输入以下命令：  
`chmod 777 /mda/07_machine_learning_classification_project/project/Geolife\ Trajectories\ 1.3/data_transformer.py`。

## 关于data_preprocessor.py的使用
该脚本针对 csv_files 文件夹下的所有csv文件进行预处理，包括计算距离、速度、加速度以及用户ID，处理之后导出到 csv_files_preprocessed 文件夹下。
