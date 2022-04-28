import os
from os import listdir

file_dir = 'E:/YGF/Yolox/YOLOX-OBB-main/YOLOX-OBB-main/result_output/val/labelTxt-v1.0/labelTxt'
file_list=listdir(file_dir)
for file in file_list:
     with open('E:/YGF/Yolox/YOLOX-OBB-main/YOLOX-OBB-main/result_output/imgnamefile/imgnamefile3.txt','a') as f:
       f.write(os.path.splitext(file)[0])
       f.write('\r\n')



# list = []
# with open('E:/YGF/Yolox/YOLOX-OBB-main/YOLOX-OBB-main/result_output/imgnamefile/imgnamefile.txt','r') as f:
#     for line in f:
#         list.append(line.strip())
# list=[x.strip()for x in list if x.strip()!='']
# print(list)
# with open('E:/YGF/Yolox/YOLOX-OBB-main/YOLOX-OBB-main/result_output/imgnamefile/imgnamefile2.txt','w') as p:
#     for item in sorted(list):
#         p.writelines(item)
#         p.writelines('\n')
