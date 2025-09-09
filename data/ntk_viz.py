import numpy as np

FFNet_data = np.load("ntk_analysis_FFNet.npy", allow_pickle=True)
# OCFFNet_data = np.load("ntk_analysis_OCFFNet.npy", allow_pickle=True)

iter_num = 0
ffnet_eigenvalues = []

item = FFNet_data[iter_num]
ffnet_eigenvalues.append(item['eigenvalue_1'])
ffnet_eigenvalues.append(item['eigenvalue_2'])
ffnet_eigenvalues.append(item['eigenvalue_3'])
ffnet_eigenvalues.append(item['eigenvalue_4'])
ffnet_eigenvalues.append(item['eigenvalue_5'])
ffnet_eigenvalues.append(item['eigenvalue_6'])
ffnet_eigenvalues.append(item['eigenvalue_7'])
ffnet_eigenvalues.append(item['eigenvalue_8'])
ffnet_eigenvalues.append(item['eigenvalue_9'])
ffnet_eigenvalues.append(item['eigenvalue_10'])


print(ffnet_eigenvalues)

# print(FFNet_data)
# print(OCFFNet_data)