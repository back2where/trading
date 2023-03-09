
import os
import pandas as pd
import glob
# path = "results"  # 指定文件夹目录,这里的data和运行的ipynb在一个文件夹中
path = "results_VMF"  # 指定文件夹目录,这里的data和运行的ipynb在一个文件夹中
files = os.listdir(path)  # 查看data文件夹下有什么文件
# print(files)
csv_list = glob.glob(os.path.join(path,'account_value_trade_ensemble_*.csv'))  # 查看同文件夹下的csv文件数
print(u'共发现%s个CSV文件' % len(csv_list))
print(u'正在处理............')
for i in csv_list: #循环读取同文件夹下的csv文件·
    fr = open(i,'rb').read()
    with open('./results_VMF/all_acount_value_trade_result.csv','ab') as f: #将结果保存为result.csv
        f.write(fr)
print('合并完毕！')
 #all = pd.read_csv('ppo/all_acount_value_trade_ppo_result.csv')
# print(all)