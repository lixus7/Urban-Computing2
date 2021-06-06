# Urban-Computing2


# github url:
**STGCN**:

pytorch: https://github.com/Aguin/STGCN-PyTorch

tensorflow: https://github.com/VeritasYin/STGCN_IJCAI-18   （Original author）

**TGCN**: 

pytorch: code in this project

tensorflow: https://github.com/lehaifeng/T-GCN             （Original author）

**Graph WaveNet**:

pytorch: https://github.com/nnzhan/Graph-WaveNet           （Original author）

**DCRNN**:

pytorch: https://github.com/chnsh/DCRNN_PyTorch

tensorflow: https://github.com/liyaguang/DCRNN             （Original author）

# download the others（data and save）
BaiduNetdisk url: 链接：https://pan.baidu.com/s/1fbIdwtW8GnyUgLPV_nnq3g 提取码：ZUYS 



there are four folders in this project

/PEMSBAY            （sava the data and some config files）

/METRLA            （sava the data and some config files）

/save            （save the model paramter and some template model files）

/workPEMSBAY     （work with METR-LA dataset）

/workMETRLA    （work with PEMS-BAY dataset）





# configuration instructions：

## 1、enviroment：
python=3.7.7

tensorflow=1.14

pytorch=1.7.1

scipy=1.5.2

numpy=1.19.1

pandas>=0.19.2

tables

statsmodels

future

pyyaml

matplotlib


# run with different dataset by working in two folder
cd workMETRLA 

or 

cd workPEMSBAY

# run STGCN....models
python STGCN.py

# run predict....
python pred*.py


 
