import functions
import os

path = ''+os.getcwd()+'/'
resultpath = path + 'result/'

# Example of one experiment.
# experiment('data/CroppedYaleB', 2, 'sp','model_L1RNMF', percentage=0.1)

# this is a set of experiments, you can just run this python file.


dataType = ['data/ORL', 'data/CroppedYaleB']
noiseTypes = ['b', 'sp']
modelTypes = ['model_stdNMF', 'model_L1NMF', 'model_L21NMF', 'model_HCNMF','model_L1RNMF']
block_sizes = [10, 12, 14]
percentages = [0.01, 0.05, 0.1]
D_list = []
R_list = []
RMSE_list = []
ACC_list = []
NMI_list = []
print(resultpath)

for data in dataType:
  for noise in noiseTypes:
    for i in range(3):
      for model in modelTypes:
        if data == 'data/ORL':
          re = 2
        elif data == 'data/CroppedYaleB':
          re = 4
        D,R,RMSE,ACC,NMI= functions.experiment(dataType=data,reduce=re,noiseType=noise,modelType=model,block_size=block_sizes[i],percentage=percentages[i])
        D_list.append(D)
        R_list.append(R)
        RMSE_list.append(RMSE)
        ACC_list.append(ACC)
        NMI_list.append(NMI)

# write the result to files

with open(resultpath+"NMI_list(1).csv", 'w') as f:
  for item in NMI_list:
    f.write("%s\n" % item)
with open(resultpath+"ACC_list(1).csv", 'w') as f:
  for item in ACC_list:
    f.write("%s\n" % item)
with open(resultpath+"RMSE_list(1).csv", 'w') as f:
  for item in RMSE_list:
    f.write("%s\n" % item)