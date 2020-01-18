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

使用方法和上基本一致。一开始会询问三个问题：
1. 检测到csv_files文件夹下数量少于用户数，是否继续。表示可供转换的文件数量并不完整，你需要确定其他文件已经转换完成。
2. 是否在转换之后自动删除csv_files内的文件。推荐删除，因为很有可能会遇到服务器内存不足的情况。若有发生，请在终端进行文件的删除操作。
3. 是否继续上一次的工作。若上一次工作被中断，可以选择继续，则检测csv_files_preprocessed文件夹下的文件，存在的则跳过。若想重新全部处理，覆盖该文件夹下的文件，则选择否。
