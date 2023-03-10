import os,math,pickle
import pandas as pd
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as pyplot


K_0 = 0.1
K_1 = 10
Alpha = 5

def CobbD_kernel(X, Y):
    k_0 =K_0
    k_1 = K_1
    alpha = Alpha
    kernel_value = k_0+np.multiply(k_1,np.power((np.dot(X, Y.T)),alpha))
    return kernel_value

def upsample(df):
		# Separate majority and minority classes
		label = str(df.columns[-1])
		df_majority = df[df[label]==1]
		df_minority = df[df[label]==0]
		# print(df_majority.shape[0],df_minority.shape[0])
		
		from sklearn.utils import  
		# Upsample minority class
		df_minority_upsampled = resample(df_minority, replace=True,     # sample with replacement
                                 n_samples=df_majority.shape[0],    # to match majority class
                                 random_state=123) # reproducible results
		df_upsampled = pd.concat([df_majority, df_minority_upsampled]) # Combine majority class with upsampled minority class
		# print(df_upsampled[label].value_counts()) # Display new class counts

		return df_upsampled




def model_run(model,df):
	n_folds = 10
	avg_tn_fp_fn_tp = np.zeros([1,4])
	avg_roc_auc = np.zeros(1)
	i = 1
	roc_auc_new =0
	filepath = './model.sav'

	# df = df.drop(df.columns[0], axis=1) # droping tcga id from first column
	df_without_tcga_id_label = df.drop([df.columns[0],df.columns[-1]], axis=1)
	X = df_without_tcga_id_label.values # dropping tcga_id and class label
	y = df[df.columns[-1]].values # storing class label in separate variable

	from sklearn.model_selection import train_test_split
	# X, X_eval, y, y_eval = train_test_split(X, y, test_size=0.1, random_state=42,stratify = y)

	from sklearn.model_selection import StratifiedKFold
	from sklearn.metrics import confusion_matrix,roc_auc_score

	skf = StratifiedKFold(n_splits=n_folds)
	for train_index, test_index in skf.split(X, y):
		print('trainig', i, 'fold')
		i+=1


		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		
		df_X_train = pd.DataFrame(X_train, columns = df_without_tcga_id_label.columns)
		df_X_train.insert(loc = len(df_X_train.columns),column = 'label',value = y_train)



		df_X_train_upsampled = upsample(df_X_train)

		X_train_upsampled = df_X_train_upsampled.drop(df_X_train_upsampled.columns[-1], axis=1).values 
		y_train_upsampled = df_X_train_upsampled[df_X_train_upsampled.columns[-1]].values # storing class label in separate variable
		
		print(df_X_train_upsampled['label'].value_counts()) # Display new class counts
		model=model.fit(X_train_upsampled, y_train_upsampled)

		# df_X_test = pd.DataFrame(X_test, columns = df_without_tcga_id_label.columns)
		# df_X_test.insert(loc = len(df_X_test.columns),column = 'label',value = y_test)
		
		# df_X_test_upsampled = upsample(df_X_test)

		# X_test_upsampled = df_X_test_upsampled.drop(df_X_test_upsampled.columns[-1], axis=1).values 
		# y_test_upsampled = df_X_test_upsampled[df_X_test_upsampled.columns[-1]].values # storing class label in separate variable
		
		
		y_test_pred=model.predict(X_test)
		tn_fp_fn_tp = confusion_matrix(y_test,y_test_pred).ravel()
		avg_tn_fp_fn_tp = avg_tn_fp_fn_tp+tn_fp_fn_tp
		roc_auc=roc_auc_score(y_test, y_test_pred)
		avg_roc_auc = avg_roc_auc+roc_auc


		if(roc_auc>roc_auc_new):
			pickle.dump(model, open(filepath, 'wb'))
			# model.save("model.h5")
			roc_auc_new=roc_auc
		model = pickle.load(open(filepath, 'rb'))
		# model = load_model('model.h5')

	cv_tn_fp_fn_tp = np.round(avg_tn_fp_fn_tp/n_folds,0)
	cv_roc_auc = avg_roc_auc/n_folds


	final_model = pickle.load(open(filepath, 'rb'))
	# final_model = load_model('model.h5')
	y_pred=final_model.predict(X)
	test_tn_fp_fn_tp = confusion_matrix(y,y_pred).ravel()
	test_roc=roc_auc_score(y,y_pred)

	return cv_roc_auc,cv_tn_fp_fn_tp,test_roc,test_tn_fp_fn_tp


cancer_type =  ['BRCA']
index = 0
os.environ["CUDA_VISIBLE_DEVICES"]="1"
path = '/Data/nikhilanand_1921cs24/VAE_SVM/'+cancer_type[index]

# df_cln = pd.read_csv(os.path.join(path,'clinical','raw_features_cln.csv'), header=0,index_col=None, delimiter=",",low_memory=False)# read the csv data file
# df_cnv = pd.read_csv(os.path.join(path,'cnv','raw_features_cnv.csv'), header=0,index_col=None, delimiter=",",low_memory=False)# read the csv data file
# df_dna = pd.read_csv(os.path.join(path,'dna_methylation','raw_features_dna.csv'), header=0,index_col=None, delimiter=",",low_memory=False)# read the csv data file
# df_mir = pd.read_csv(os.path.join(path,'miRSeq','raw_features_mir.csv'), header=0,index_col=None, delimiter=",",low_memory=False)# read the csv data file
# df_mrna = pd.read_csv(os.path.join(path,'mRNA','raw_features_mrna.csv'), header=0,index_col=None, delimiter=",",low_memory=False)# read the csv data file
# df_wsi = pd.read_csv(os.path.join(path,'wsi','raw_features_wsi.csv'), header=0,index_col=None, delimiter=",",low_memory=False)# read the csv data file



# from sklearn.preprocessing import MinMaxScaler
# min_max_scaler = MinMaxScaler()
# df_cln[["age_at_initial_pathologic_diagnosis"]] = min_max_scaler.fit_transform(df_cln[["age_at_initial_pathologic_diagnosis"]])


# df_cln = pd.read_csv(os.path.join(path,'clinical','pca_features_cln.csv'), header=0,index_col=None, delimiter=",",low_memory=False)# read the csv data file
# df_cnv = pd.read_csv(os.path.join(path,'cnv','pca_features_cnv.csv'), header=0,index_col=None, delimiter=",",low_memory=False)# read the csv data file
# df_dna = pd.read_csv(os.path.join(path,'dna_methylation','pca_features_dna.csv'), header=0,index_col=None, delimiter=",",low_memory=False)# read the csv data file
# df_mir = pd.read_csv(os.path.join(path,'miRSeq','pca_features_mir.csv'), header=0,index_col=None, delimiter=",",low_memory=False)# read the csv data file
# df_mrna = pd.read_csv(os.path.join(path,'mRNA','pca_features_mrna.csv'), header=0,index_col=None, delimiter=",",low_memory=False)# read the csv data file
# df_wsi = pd.read_csv(os.path.join(path,'wsi','pca_features_wsi.csv'), header=0,index_col=None, delimiter=",",low_memory=False)# read the csv data file


df_cln = pd.read_csv(os.path.join(path,'clinical','vae_features_cln.csv'), header=0,index_col=None, delimiter=",",low_memory=False)# read the csv data file
df_cnv = pd.read_csv(os.path.join(path,'cnv','vae_features_cnv.csv'), header=0,index_col=None, delimiter=",",low_memory=False)# read the csv data file
df_dna = pd.read_csv(os.path.join(path,'dna_methylation','vae_features_dna.csv'), header=0,index_col=None, delimiter=",",low_memory=False)# read the csv data file
df_mir = pd.read_csv(os.path.join(path,'miRSeq','vae_features_mir.csv'), header=0,index_col=None, delimiter=",",low_memory=False)# read the csv data file
df_mrna = pd.read_csv(os.path.join(path,'mRNA','vae_features_mrna.csv'), header=0,index_col=None, delimiter=",",low_memory=False)# read the csv data file
df_wsi = pd.read_csv(os.path.join(path,'wsi','vae_features_wsi.csv'), header=0,index_col=None, delimiter=",",low_memory=False)# read the csv data file











raw_modalities = ['cln','cnv','dna','mir','mrna','wsi']
raw_dfs = [df_cln,df_cnv,df_dna,df_mir,df_mrna,df_wsi]
number_of_modalities = 6


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# training uni-modal based machine learning classifiers#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
def uni_ml():
	for i in range(0,number_of_modalities):
		df = raw_dfs[i]

		print(df.shape)
		from sklearn.svm import SVC
		import csv
		for kernel in ['rbf','linear','poly','sigmoid']:
			model = SVC(kernel=kernel)
			roc_confusion_matrix = model_run(model,df)

			final_results = [raw_modalities[i],kernel,
						str(roc_confusion_matrix[0][0]),
						str(roc_confusion_matrix[1][0][0]),
						str(roc_confusion_matrix[1][0][1]),
						str(roc_confusion_matrix[1][0][2]),
						str(roc_confusion_matrix[1][0][3]),
						str(roc_confusion_matrix[2]),
						str(roc_confusion_matrix[3][0]),
						str(roc_confusion_matrix[3][1]),
						str(roc_confusion_matrix[3][2]),
						str(roc_confusion_matrix[3][3])]
			print(final_results)
			with open('./uni_modal_results.csv', 'a') as f:
				writer = csv.writer(f)
				writer.writerow(final_results)


		import csv
		from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
		model = RandomForestClassifier(n_estimators=10, criterion= 'gini', max_features = None)
		# model = AdaBoostClassifier(base_estimator=model,n_estimators=200) 
		# model = SVC(C=1,kernel=CobbD_kernel)
		roc_confusion_matrix = model_run(model,df)
		final_results = [raw_modalities[i],'RF',
						str(roc_confusion_matrix[0][0]),
						str(roc_confusion_matrix[1][0][0]),
						str(roc_confusion_matrix[1][0][1]),
						str(roc_confusion_matrix[1][0][2]),
						str(roc_confusion_matrix[1][0][3]),
						str(roc_confusion_matrix[2]),
						str(roc_confusion_matrix[3][0]),
						str(roc_confusion_matrix[3][1]),
						str(roc_confusion_matrix[3][2]),
						str(roc_confusion_matrix[3][3])]

		print(final_results)
		with open('./uni_modal_results.csv', 'a') as f:
			writer = csv.writer(f)
			writer.writerow(final_results)

		

		
		# from sklearn.ensemble import RandomForestRegressor
		
		# regr = RandomForestRegressor(n_estimators=1000, criterion= 'squared_error', max_features = None, max_depth=None, random_state=0,n_jobs=-1)
		# regr.fit(X, y)
		# predicted = regr.predict(X)
		# # print(regr.predict(X))
		# from sklearn.metrics import mean_squared_error
		# errors = mean_squared_error(y, predicted, squared=False)
		# print(errors)
		# from sklearn.metrics import r2_score
		# r2 = r2_score(y,predicted)
		# print(r2)	
		# # calculate errors
		# errors = list()
		# for i in range(len(y)):
		# 	# calculate error
		# 	err = (y[i] - predicted[i])**2
		# 	# store error
		# 	errors.append(err)
		# 	# report error
		# 	# print('>%.1f, %.1f = %.3f' % (y[i], predicted[i], err))
		# # plot errors
		# pyplot.plot(errors)
		# # pyplot.xticks(ticks=[i for i in range(len(errors))], labels=predicted)
		# pyplot.xlabel('Predicted Value')
		# pyplot.ylabel('Mean Squared Error')
		# pyplot.savefig('./regression.png')


		

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# training Bi-modal based machine learning classifiers#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def bi_ml():
	for i in range(0,number_of_modalities):
		for j in range(i+1,number_of_modalities):
			df1 = raw_dfs[i]
			df2 = raw_dfs[j]
			# print(df1)
			# print(df2)
			# selcting the common patients between clinical and CNV
			df1_df2 = pd.merge(df1, df2, how ='inner', left_on =df1.columns[0], right_on =df2.columns[0])
			# two labels label_x and label_y has been included in the final dataframe corresponding to clinical and CNV, dropping the label_x
			df1_df2 = df1_df2.drop('label_'+str(raw_modalities[i]), axis=1)
			print(raw_modalities[i]+'_'+raw_modalities[j],df1_df2.shape[0])

			print(raw_modalities[i],raw_modalities[j])
			# df1_df2 = df1_df2.drop(df1_df2.columns[0], axis=1) # droping tcga id from first column
			# X = df1_df2.drop(df1_df2.columns[len(df1_df2.columns)-1], axis=1).values # dropping class label from last column
			# y = df1_df2[df1_df2.columns[len(df1_df2.columns)-1]].values # storing class label in separate variable
			# print(X.shape)
			# print(y.shape)

			from sklearn.svm import SVC
			import csv
			for kernel in ['rbf','linear','poly','sigmoid']:
				model = SVC(C=10, kernel=kernel,gamma='auto')
				roc_confusion_matrix = model_run(model,df1_df2)
				final_results = [raw_modalities[i]+'_'+raw_modalities[j],kernel,
									str(roc_confusion_matrix[0][0]),
									str(roc_confusion_matrix[1][0][0]),
									str(roc_confusion_matrix[1][0][1]),
									str(roc_confusion_matrix[1][0][2]),
									str(roc_confusion_matrix[1][0][3]),
									str(roc_confusion_matrix[2]),
									str(roc_confusion_matrix[3][0]),
									str(roc_confusion_matrix[3][1]),
									str(roc_confusion_matrix[3][2]),
									str(roc_confusion_matrix[3][3])]

				with open('./bi_modal_results.csv', 'a') as f:
					writer = csv.writer(f)
					writer.writerow(final_results)
			

			from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
			model = RandomForestClassifier(n_estimators=1000, criterion= 'gini', max_features = None, max_depth=None, random_state=0,class_weight='balanced',n_jobs=-1)

			# model = AdaBoostClassifier(base_estimator=model,n_estimators=200) 
			roc_confusion_matrix = model_run(model,df1_df2)
			final_results = [raw_modalities[i]+'_'+raw_modalities[j],'RF',
									str(roc_confusion_matrix[0][0]),
									str(roc_confusion_matrix[1][0][0]),
									str(roc_confusion_matrix[1][0][1]),
									str(roc_confusion_matrix[1][0][2]),
									str(roc_confusion_matrix[1][0][3]),
									str(roc_confusion_matrix[2]),
									str(roc_confusion_matrix[3][0]),
									str(roc_confusion_matrix[3][1]),
									str(roc_confusion_matrix[3][2]),
									str(roc_confusion_matrix[3][3])]
			print(final_results)
			import csv
			with open('./bi_modal_results.csv', 'a') as f:
				writer = csv.writer(f)
				writer.writerow(final_results)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# training Tri-modal based machine learning classifiers#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
def tri_ml():
	for i in range(0,number_of_modalities):
		for j in range(i+1,number_of_modalities):
			for k in range(j+1,number_of_modalities):
				df1 = raw_dfs[i]
				df2 = raw_dfs[j]
				df3 = raw_dfs[k]
				# print(df1)
				# print(df2)
				# selcting the common patients between clinical and CNV
				df1_df2 = pd.merge(df1, df2, how ='inner', left_on =df1.columns[0], right_on =df2.columns[0])
				# two labels label_x and label_y has been included in the final dataframe corresponding to clinical and CNV, dropping the label_x
				# df1_df2 = df1_df2.drop('label_x', axis=1)

				df1_df2_df3 = pd.merge(df1_df2,df3, how ='inner', left_on =df1_df2.columns[0], right_on =df3.columns[0])
				
				
				df1_df2_df3 = df1_df2_df3.drop(['label_'+str(raw_modalities[i]),'label_'+str(raw_modalities[j])], axis=1)
				# print(df1_df2_df3)
				# print(raw_modalities[i],raw_modalities[j],raw_modalities[k])
				# df1_df2_df3 = df1_df2_df3.drop(df1_df2_df3.columns[0], axis=1) # droping tcga id from first column
				# X = df1_df2_df3.drop(df1_df2_df3.columns[len(df1_df2_df3.columns)-1], axis=1).values # dropping class label from last column
				# y = df1_df2_df3[df1_df2_df3.columns[len(df1_df2_df3.columns)-1]].values # storing class label in separate variable
				# # print(X.shape)
				# print(y.shape)
				# print(df1_df2_df3)
				print(raw_modalities[i]+'_'+raw_modalities[j]+'_'+raw_modalities[k],df1_df2_df3.shape[0])

				from sklearn.svm import SVC
				import csv
				for kernel in ['rbf','linear','poly','sigmoid']:
					model = SVC(C=10, kernel=kernel,gamma='auto')
					roc_confusion_matrix = model_run(model,df1_df2_df3)
					final_results = [raw_modalities[i]+'_'+raw_modalities[j]+'_'+raw_modalities[k],kernel,
									str(roc_confusion_matrix[0][0]),
									str(roc_confusion_matrix[1][0][0]),
									str(roc_confusion_matrix[1][0][1]),
									str(roc_confusion_matrix[1][0][2]),
									str(roc_confusion_matrix[1][0][3]),
									str(roc_confusion_matrix[2]),
									str(roc_confusion_matrix[3][0]),
									str(roc_confusion_matrix[3][1]),
									str(roc_confusion_matrix[3][2]),
									str(roc_confusion_matrix[3][3])]
					print(final_results)
					with open('./tri_modal_results.csv', 'a') as f:
						writer = csv.writer(f)
						writer.writerow(final_results)
				

				from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
				model = RandomForestClassifier(n_estimators=1000, criterion= 'gini', max_features = None, max_depth=None, random_state=0,class_weight='balanced',n_jobs=-1)
				# model = AdaBoostClassifier(base_estimator=model,n_estimators=200) 
				roc_confusion_matrix = model_run(model,df1_df2_df3)
				final_results = [raw_modalities[i]+'_'+raw_modalities[j]+'_'+raw_modalities[k],'RF',
								str(roc_confusion_matrix[0][0]),
								str(roc_confusion_matrix[1][0][0]),
								str(roc_confusion_matrix[1][0][1]),
								str(roc_confusion_matrix[1][0][2]),
								str(roc_confusion_matrix[1][0][3]),
								str(roc_confusion_matrix[2]),
								str(roc_confusion_matrix[3][0]),
								str(roc_confusion_matrix[3][1]),
								str(roc_confusion_matrix[3][2]),
								str(roc_confusion_matrix[3][3])]
				print(final_results)
				import csv
				with open('./tri_modal_results.csv', 'a') as f:
					writer = csv.writer(f)
					writer.writerow(final_results)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# training Quad-modal based machine learning classifiers#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
def quad_ml():
	for i in range(0,number_of_modalities):
		for j in range(i+1,number_of_modalities):
			for k in range(j+1,number_of_modalities):
				for l in range(k+1,number_of_modalities):
					df1 = raw_dfs[i]
					df2 = raw_dfs[j]
					df3 = raw_dfs[k]
					df4 = raw_dfs[l]
					# print(df1)
					# print(df2)
					# selcting the common patients between clinical and CNV
					df1_df2 = pd.merge(df1, df2, how ='inner', left_on =df1.columns[0], right_on =df2.columns[0])
					# two labels label_x and label_y has been included in the final dataframe corresponding to clinical and CNV, dropping the label_x
					# df1_df2 = df1_df2.drop('label_x', axis=1)

					df1_df2_df3 = pd.merge(df1_df2,df3, how ='inner', left_on =df1_df2.columns[0], right_on =df3.columns[0])
					# df1_df2_df3 = df1_df2_df3.drop('label', axis=1)

					df1_df2_df3_df4 = pd.merge(df1_df2_df3,df4, how ='inner', left_on =df1_df2_df3.columns[0], right_on =df4.columns[0])
					df1_df2_df3_df4 = df1_df2_df3_df4.drop(['label_'+str(raw_modalities[i]),
															'label_'+str(raw_modalities[j]),
															'label_'+str(raw_modalities[k])], axis=1)
					
					# print(raw_modalities[i],raw_modalities[j])
					# df1_df2_df3_df4 = df1_df2_df3_df4.drop(df1_df2_df3_df4.columns[0], axis=1) # droping tcga id from first column
					# X = df1_df2_df3_df4.drop(df1_df2_df3_df4.columns[len(df1_df2_df3_df4.columns)-1], axis=1).values # dropping class label from last column
					# y = df1_df2_df3_df4[df1_df2_df3_df4.columns[len(df1_df2_df3_df4.columns)-1]].values # storing class label in separate variable
					# print(X.shape)
					# print(y.shape)
					print(raw_modalities[i]+'_'+raw_modalities[j]+'_'+raw_modalities[k]+'_'+raw_modalities[l],df1_df2_df3_df4.shape[0])
					from sklearn.svm import SVC
					import csv
					for kernel in ['rbf','linear','poly','sigmoid']:
						model = SVC(C=10, kernel=kernel,gamma='auto')
						roc_confusion_matrix = model_run(model,df1_df2_df3_df4)
						final_results = [raw_modalities[i]+'_'+raw_modalities[j]+'_'+raw_modalities[k]+'_'+raw_modalities[l],kernel,
										str(roc_confusion_matrix[0][0]),
										str(roc_confusion_matrix[1][0][0]),
										str(roc_confusion_matrix[1][0][1]),
										str(roc_confusion_matrix[1][0][2]),
										str(roc_confusion_matrix[1][0][3]),
										str(roc_confusion_matrix[2]),
										str(roc_confusion_matrix[3][0]),
										str(roc_confusion_matrix[3][1]),
										str(roc_confusion_matrix[3][2]),
										str(roc_confusion_matrix[3][3])]
						print(final_results)
						with open('./quad_modal_results.csv', 'a') as f:
							writer = csv.writer(f)
							writer.writerow(final_results)
					

					from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
					clf = RandomForestClassifier(n_estimators=1000, criterion= 'gini', max_features = None, max_depth=None, random_state=0,class_weight='balanced',n_jobs=-1)

					model = AdaBoostClassifier(base_estimator=clf,n_estimators=200) 
					roc_confusion_matrix = model_run(model,df1_df2_df3_df4)
					final_results = [raw_modalities[i]+'_'+raw_modalities[j]+'_'+raw_modalities[k]+'_'+raw_modalities[l],'RF',
										str(roc_confusion_matrix[0][0]),
										str(roc_confusion_matrix[1][0][0]),
										str(roc_confusion_matrix[1][0][1]),
										str(roc_confusion_matrix[1][0][2]),
										str(roc_confusion_matrix[1][0][3]),
										str(roc_confusion_matrix[2]),
										str(roc_confusion_matrix[3][0]),
										str(roc_confusion_matrix[3][1]),
										str(roc_confusion_matrix[3][2]),
										str(roc_confusion_matrix[3][3])]

					print(final_results)
					import csv
					with open('./quad_modal_results.csv', 'a') as f:
						writer = csv.writer(f)
						writer.writerow(final_results)



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# training penta-modal based machine learning classifiers#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
def penta_ml():
	for i in range(0,number_of_modalities):
		for j in range(i+1,number_of_modalities):
			for k in range(j+1,number_of_modalities):
				for l in range(k+1,number_of_modalities):
					for m in range(l+1,number_of_modalities):
						df1 = raw_dfs[i]
						df2 = raw_dfs[j]
						df3 = raw_dfs[k]
						df4 = raw_dfs[l]
						df5 = raw_dfs[m]
						# print(df1)
						# print(df2)
						# selcting the common patients between clinical and CNV
						df1_df2 = pd.merge(df1, df2, how ='inner', left_on =df1.columns[0], right_on =df2.columns[0])
						# two labels label_x and label_y has been included in the final dataframe corresponding to clinical and CNV, dropping the label_x
						# df1_df2 = df1_df2.drop('label_x', axis=1)

						df1_df2_df3 = pd.merge(df1_df2,df3, how ='inner', left_on =df1_df2.columns[0], right_on =df3.columns[0])
						# df1_df2_df3 = df1_df2_df3.drop('label', axis=1)

						df1_df2_df3_df4 = pd.merge(df1_df2_df3,df4, how ='inner', left_on =df1_df2_df3.columns[0], right_on =df4.columns[0])
						# df1_df2_df3_df4 = df1_df2_df3_df4.drop('label', axis=1)

						df1_df2_df3_df4_df5 = pd.merge(df1_df2_df3_df4,df5, how ='inner', left_on =df1_df2_df3_df4.columns[0], right_on =df5.columns[0])
						df1_df2_df3_df4_df5 = df1_df2_df3_df4_df5.drop(['label_'+str(raw_modalities[i]),
																		'label_'+str(raw_modalities[j]),
																		'label_'+str(raw_modalities[k]),
																		'label_'+str(raw_modalities[l])], axis=1)
						
						# print(raw_modalities[i],raw_modalities[j])
						# df1_df2_df3_df4_df5 = df1_df2_df3_df4_df5.drop(df1_df2_df3_df4_df5.columns[0], axis=1) # droping tcga id from first column
						# X = df1_df2_df3_df4_df5.drop(df1_df2_df3_df4_df5.columns[len(df1_df2_df3_df4_df5.columns)-1], axis=1).values # dropping class label from last column
						# y = df1_df2_df3_df4_df5[df1_df2_df3_df4_df5.columns[len(df1_df2_df3_df4_df5.columns)-1]].values # storing class label in separate variable
						# print(X.shape)
						# print(y.shape)
						print(raw_modalities[i]+'_'+raw_modalities[j]+'_'+raw_modalities[k]+'_'+raw_modalities[l]+'_'+raw_modalities[m],df1_df2_df3_df4_df5.shape[0])

						from sklearn.svm import SVC
						import csv
						for kernel in ['rbf','linear','poly','sigmoid']:
							model = SVC(C=10, kernel=kernel,gamma='auto')
							roc_confusion_matrix = model_run(model,df1_df2_df3_df4_df5)
							final_results = [raw_modalities[i]+'_'+raw_modalities[j]+'_'+raw_modalities[k]+'_'+raw_modalities[l]+'_'+raw_modalities[m],kernel,
										str(roc_confusion_matrix[0][0]),
										str(roc_confusion_matrix[1][0][0]),
										str(roc_confusion_matrix[1][0][1]),
										str(roc_confusion_matrix[1][0][2]),
										str(roc_confusion_matrix[1][0][3]),
										str(roc_confusion_matrix[2]),
										str(roc_confusion_matrix[3][0]),
										str(roc_confusion_matrix[3][1]),
										str(roc_confusion_matrix[3][2]),
										str(roc_confusion_matrix[3][3])]
							print(final_results)
							with open('./penta_modal_results.csv', 'a') as f:
								writer = csv.writer(f)
								writer.writerow(final_results)
						

						from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
						model = RandomForestClassifier(n_estimators=1000, criterion= 'gini', max_features = None, max_depth=None, random_state=0,class_weight='balanced',n_jobs=-1)

						# model = AdaBoostClassifier(base_estimator=clf,n_estimators=200) 
						roc_confusion_matrix = model_run(model,df1_df2_df3_df4_df5)
						final_results = [raw_modalities[i]+'_'+raw_modalities[j]+'_'+raw_modalities[k]+'_'+raw_modalities[l]+'_'+raw_modalities[m],'RF',
										str(roc_confusion_matrix[0][0]),
										str(roc_confusion_matrix[1][0][0]),
										str(roc_confusion_matrix[1][0][1]),
										str(roc_confusion_matrix[1][0][2]),
										str(roc_confusion_matrix[1][0][3]),
										str(roc_confusion_matrix[2]),
										str(roc_confusion_matrix[3][0]),
										str(roc_confusion_matrix[3][1]),
										str(roc_confusion_matrix[3][2]),
										str(roc_confusion_matrix[3][3])]
						print(final_results)
						import csv
						with open('./penta_modal_results.csv', 'a') as f:
							writer = csv.writer(f)
							writer.writerow(final_results)



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# training hexa-modal based machine learning classifiers#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
def hexa_ml():
	for i in range(0,number_of_modalities):
		for j in range(i+1,number_of_modalities):
			for k in range(j+1,number_of_modalities):
				for l in range(k+1,number_of_modalities):
					for m in range(l+1,number_of_modalities):
						for n in range(m+1,number_of_modalities):
							df1 = raw_dfs[i]
							df2 = raw_dfs[j]
							df3 = raw_dfs[k]
							df4 = raw_dfs[l]
							df5 = raw_dfs[m]
							df6 = raw_dfs[n]
							# print(df1)
							# print(df2)
							# selcting the common patients between clinical and CNV
							df1_df2 = pd.merge(df1, df2, how ='inner', left_on =df1.columns[0], right_on =df2.columns[0])
							# two labels label_x and label_y has been included in the final dataframe corresponding to clinical and CNV, dropping the label_x
							# df1_df2 = df1_df2.drop('label_x', axis=1)

							df1_df2_df3 = pd.merge(df1_df2,df3, how ='inner', left_on =df1_df2.columns[0], right_on =df3.columns[0])
							# df1_df2_df3 = df1_df2_df3.drop('label', axis=1)

							df1_df2_df3_df4 = pd.merge(df1_df2_df3,df4, how ='inner', left_on =df1_df2_df3.columns[0], right_on =df4.columns[0])
							# df1_df2_df3_df4 = df1_df2_df3_df4.drop('label', axis=1)

							df1_df2_df3_df4_df5 = pd.merge(df1_df2_df3_df4,df5, how ='inner', left_on =df1_df2_df3_df4.columns[0], right_on =df5.columns[0])
							# df1_df2_df3_df4_df5 = df1_df2_df3_df4_df5.drop('label', axis=1)
							
							df1_df2_df3_df4_df5_df6 = pd.merge(df1_df2_df3_df4_df5,df6, how ='inner', left_on =df1_df2_df3_df4_df5.columns[0], right_on =df6.columns[0])
							df1_df2_df3_df4_df5_df6 = df1_df2_df3_df4_df5_df6.drop(['label_'+str(raw_modalities[i]),
																					'label_'+str(raw_modalities[j]),
																					'label_'+str(raw_modalities[k]),
																					'label_'+str(raw_modalities[l]),
																					'label_'+str(raw_modalities[m])], axis=1)

							# print(raw_modalities[i],raw_modalities[j])
							# df1_df2_df3_df4_df5_df6 = df1_df2_df3_df4_df5_df6.drop(df1_df2_df3_df4_df5_df6.columns[0], axis=1) # droping tcga id from first column
							# X = df1_df2_df3_df4_df5_df6.drop(df1_df2_df3_df4_df5_df6.columns[len(df1_df2_df3_df4_df5_df6.columns)-1], axis=1).values # dropping class label from last column
							# y = df1_df2_df3_df4_df5_df6[df1_df2_df3_df4_df5_df6.columns[len(df1_df2_df3_df4_df5_df6.columns)-1]].values # storing class label in separate variable
							# print(X.shape)
							# print(y.shape)
							print(raw_modalities[i]+'_'+raw_modalities[j]+'_'+raw_modalities[k]+'_'+raw_modalities[l]+'_'+raw_modalities[m]+'_'+raw_modalities[n],df1_df2_df3_df4_df5_df6.shape[0])

							from sklearn.svm import SVC
							import csv
							for kernel in ['rbf','linear','poly','sigmoid']:
								model = SVC(C=10, kernel=kernel,gamma='auto')
								roc_confusion_matrix = model_run(model,df1_df2_df3_df4_df5_df6)
								final_results = [raw_modalities[i]+'_'+raw_modalities[j]+'_'+raw_modalities[k]+'_'+raw_modalities[l]+'_'+raw_modalities[m]+'_'+raw_modalities[n],kernel,
										str(roc_confusion_matrix[0][0]),
										str(roc_confusion_matrix[1][0][0]),
										str(roc_confusion_matrix[1][0][1]),
										str(roc_confusion_matrix[1][0][2]),
										str(roc_confusion_matrix[1][0][3]),
										str(roc_confusion_matrix[2]),
										str(roc_confusion_matrix[3][0]),
										str(roc_confusion_matrix[3][1]),
										str(roc_confusion_matrix[3][2]),
										str(roc_confusion_matrix[3][3])]
								print(final_results)
								with open('./hexa_modal_results.csv', 'a') as f:
									writer = csv.writer(f)
									writer.writerow(final_results)
							

							from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
							model = RandomForestClassifier(n_estimators=1000, criterion= 'gini', max_features = None, max_depth=None, random_state=0,class_weight='balanced',n_jobs=-1)

							# model = AdaBoostClassifier(base_estimator=model,n_estimators=200) 
							roc_confusion_matrix = model_run(model,df1_df2_df3_df4_df5_df6)
							final_results = [raw_modalities[i]+'_'+raw_modalities[j]+'_'+raw_modalities[k]+'_'+raw_modalities[l]+'_'+raw_modalities[m]+'_'+raw_modalities[n],'RF',
										str(roc_confusion_matrix[0][0]),
										str(roc_confusion_matrix[1][0][0]),
										str(roc_confusion_matrix[1][0][1]),
										str(roc_confusion_matrix[1][0][2]),
										str(roc_confusion_matrix[1][0][3]),
										str(roc_confusion_matrix[2]),
										str(roc_confusion_matrix[3][0]),
										str(roc_confusion_matrix[3][1]),
										str(roc_confusion_matrix[3][2]),
										str(roc_confusion_matrix[3][3])]
							print(final_results)
							import csv
							with open('./hexa_modal_results.csv', 'a') as f:
								writer = csv.writer(f)
								writer.writerow(final_results)


uni_ml()
bi_ml()
tri_ml()
quad_ml()
penta_ml()
hexa_ml()