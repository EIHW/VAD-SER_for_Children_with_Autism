import os 
import sklearn
import numpy as np 
import pandas as pd 
import glob 
import sys
from sklearn.preprocessing import MinMaxScaler
import logging
logging.basicConfig(level = logging.INFO)
partitions = ['train','devel','test']

feature_path = '../features/'

train_list = []
train_filename = []
devel_list = []
devel_filename = []
test_list = [] 
test_filename = []

for part in partitions:
	print(part,flush=True)
	sys.stdout.flush()
 
	if part == 'train':
		df = pd.read_csv('../train.csv')
	if part == 'devel':
		df = pd.read_csv('../devel.csv')	
	if part == 'test':
		df = pd.read_csv('../test.csv')
	# ~ count = 0 
	for index, row in df.iterrows():
		session_type = row['session']
		print(session_type,flush=True)
		sys.stdout.flush()
		
		for file in glob.glob(feature_path + session_type + '*csv'):
			# ~ print(file)
			feature_file = pd.read_csv(file, sep=';', header='infer', index_col=None, usecols=range(1,130+1), dtype=np.float32)
			filename_row = pd.read_csv(file, sep=';')['name']
			if part == 'train':
				train_list.append(feature_file)
				train_filename.append(filename_row)
			if part == 'devel':
				devel_list.append(feature_file)
				devel_filename.append(filename_row)
			if part == 'test':
				test_list.append(feature_file)
				test_filename.append(filename_row)

X_train		= pd.concat(train_list, axis=0).reset_index()
y_train		= pd.concat(train_filename, axis=0).reset_index()
X_devel		= pd.concat(devel_list, axis=0).reset_index()
y_devel		= pd.concat(devel_filename, axis=0).reset_index()
X_test		= pd.concat(test_list, axis=0).reset_index()
y_test		= pd.concat(test_filename, axis=0).reset_index()

print('scaling',flush=True)
sys.stdout.flush()
scaler       = MinMaxScaler()
X_train      = scaler.fit_transform(X_train)
X_devel      = scaler.transform(X_devel)
X_test       = scaler.transform(X_test)

#X_train = pd.DataFrame(X_train)
X_devel = pd.DataFrame(X_devel)
X_test = pd.DataFrame(X_test)

print('put back together',flush=True)
sys.stdout.flush()
#train_all	= pd.concat([y_train,X_train], axis=1).reset_index()
devel_all	= pd.concat([y_devel,X_devel], axis=1).reset_index()
test_all	= pd.concat([y_test,X_test], axis=1).reset_index()

#train_all = train_all.iloc[:, 2:]
devel_all = devel_all.iloc[:, 2:]
test_all = test_all.iloc[:, 2:]


#print('saving large train csv',flush=True)
#sys.stdout.flush()
#train_all.to_csv('largetrain.csv',index=False)
print('saving large devel csv',flush=True)
sys.stdout.flush()
devel_all.to_csv('largedevel.csv',index=False)
print('saving large test csv',flush=True)
sys.stdout.flush()
test_all.to_csv('largetest.csv',index=False)