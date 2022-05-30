
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import resample
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.utils import resample
from sklearn import preprocessing

def load_data(data_name):   

	if data_name == "Jdata/dbpedia":
		features, target = datasets.load_svmlight_file("Data/Jdata/dbpedia_train.svm")

	df = pd.read_csv(f'./Data/{data_name}.csv')
	features = np.array(df.drop("Class", axis=1))
	target = np.array(df["Class"])

	le = preprocessing.LabelEncoder()
	le.fit(target)
	target = le.transform(target)

	if "cifar10" == data_name or "mnist" in data_name:
		features = features.astype('float32')
		features /= 255

	if "digits" == data_name:
		features = features.astype('float32')
		features /= 16
	if "cifar10small" == data_name:
		features = features.astype('float32')
		features /= 256

	return features, target

def split_data(features, target, split, seed=1):
   x_train, x_test, y_train, y_test = train_test_split(features,target,test_size=split, shuffle=True, random_state=seed, stratify=target)
   return x_train, x_test, y_train, y_test

def balance_dataset(df):
	# Separate majority and minority classes
	y = df.Class.unique()
	df_loss = df[df.Class==y[0]]
	df_win  = df[df.Class==y[1]]
	
	max_len = len(df_loss)
	if len(df_win) > max_len:
		max_len = len(df_win)
	# Upsample minority class
	df_upsampled_win = resample(df_win, 
                                replace=True,           # sample with replacement
                                n_samples=max_len,      # to match majority class
                                random_state=123)       # reproducible results
	 
	# Combine majority class with upsampled minority class
	df_balance = pd.concat([df_loss, df_upsampled_win])
	return df_balance

def load_arff_data(name="adult", convert_to_int=True, type="binary", log=True):
	dataset = arff.load(open(f'/home/mhshaker/projects/uncertainty/Data/{name}.arff', 'r'))
	df = pd.DataFrame(dataset['data'])
	df = df.sample(frac=1).reset_index(drop=True)
	df.rename(columns={ df.columns[-1]: "target" }, inplace = True)
	if log:
		print(f"Data name = {name}")
		print(df.head())
		print(df['target'].value_counts())

	if convert_to_int:
		for column in df:
			df[column] = df[column].astype("category").cat.codes
	# print(df.head())
	# exit()
	features = df.drop("target", axis=1)
	# print(features)
	features = preprocessing.scale(features)
	# print(features)

	# features_names = list(features.columns)
	# target_names = ["class1","class2"]

	return features, df.target #features_names, target_names