import os,math,pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



cancer_type =  ['BRCA']
index = 0
os.environ["CUDA_VISIBLE_DEVICES"]="1"
path = '/Data/nikhilanand_1921cs24/VAE_SVM/'+cancer_type[index]

def cln_raw_feature():
	cln_file_name = 'BRCA_cln_categorical.csv'

	# transposing raw cln data and sorting them by patietnt Ids or TCGA Ids
	# =============================================================================
	df_cln = pd.read_csv(os.path.join(path,'clinical',cln_file_name), header=0,index_col=None, delimiter=",",low_memory=False)# read the csv data file
	# df_cln.drop_duplicates(subset ='submitter_id.samples',keep='first',inplace=True)
	df_cln=df_cln.sort_values(by=['submitter_id.samples'],ascending=True) # sorting based on tcga ids

	# =============================================================================
	#Mapping cln patient ids with survival labels
	# =============================================================================
	df_cln_id=df_cln['submitter_id.samples'] # extracting tcga ids from cln dataframe
	df_cln_id = df_cln_id.str[:-1]
	df_cln = df_cln.drop('submitter_id.samples', axis=1) # remving old tcga ids from data
	df_cln.insert(0, "submitter_id.samples", df_cln_id.values, True) #adding the modified tcga_id
	df_survival_label = pd.read_csv(os.path.join(path,'survival_class','5_year_survival.csv'),delimiter=',') # reading survival csv file
	df_survival_label.drop_duplicates(subset ='submitter_id.samples',keep='first',inplace=True)
	survival_df = df_survival_label[df_survival_label['submitter_id.samples'].isin(df_cln_id)] # selcting survival label of tcga ids matching with cln tcga ids
	survival_df=survival_df.sort_values(by=['submitter_id.samples'],ascending=True) # sorting based on tcga ids
	survival_id = survival_df['submitter_id.samples'] # extracting tcga ids from survival labels
	df_cln = df_cln[df_cln['submitter_id.samples'].isin(survival_id)] # selecting cln data of patients matching with avivalable survival ids
	df_cln.drop_duplicates(subset ='submitter_id.samples',keep='first',inplace=True)
	tcga_id = df_cln['submitter_id.samples']
	#get all categorical columns
	cat_columns = df_cln.select_dtypes(['object']).columns

	#convert all categorical columns to numeric
	df_cln[cat_columns] = df_cln[cat_columns].apply(lambda x: pd.factorize(x)[0])
	class_labels = survival_df['survival_years'] # fetching the class labels
	df_cln=df_cln.drop(df_cln.columns[0], axis = 1) # removing the submitter_id.samples column as patients ids has been transformed to numeric values
	
	df_cln.insert(loc = 0,column = 'submitter_id.samples',value = tcga_id.values)	# re-inserting the submitter_id.samples
	df_cln.insert(loc = len(df_cln.columns),column = 'label_cln',value = class_labels.values)	
	print(df_cln)
	return df_cln

def cnv_raw_feature():
	# transposing raw CNV data and sorting them by patietnt Ids or TCGA Ids
	# =============================================================================
	cnv_file_name = 'Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes'
	df_cnv = pd.read_csv(os.path.join(path,'cnv',cnv_file_name), header=None,index_col=0, delimiter="\t",low_memory=False).T# read the csv data file and transpose it
	df_cnv=df_cnv.sort_values(by=['Gene Symbol'],ascending=True) # sorting based on tcga ids
	df_cnv.drop_duplicates(subset ="Gene Symbol",keep='first',inplace=True)
	# =============================================================================
	#Mapping CNV patient ids with survival labels
	# =============================================================================
	df_cnv_id=df_cnv['Gene Symbol'] # extracting tcga ids from CNV dataframe
	df_survival_label = pd.read_csv(os.path.join(path,'survival_class','5_year_survival.csv'),delimiter=',') # reading survival csv file
	survival_df = df_survival_label[df_survival_label['submitter_id.samples'].isin(df_cnv_id)] # selcting survival label of tcga ids matching with CNV tcga ids
	survival_df=survival_df.sort_values(by=['submitter_id.samples'],ascending=True) # sorting survival labels based on tcga ids
	survival_df.drop_duplicates(subset ="submitter_id.samples",keep='first',inplace=True)
	survival_id = survival_df['submitter_id.samples'] # extracting tcga ids from survival labels
	df_cnv = df_cnv[df_cnv['Gene Symbol'].isin(survival_id)] # selecting CNV data of patients matching with avivalable survival ids
	df_cnv.drop_duplicates(subset ="Gene Symbol",keep='first',inplace=True)
	class_labels = survival_df['5_year_cutoff'] # fetching the class labels
	tcga_id = df_cnv['Gene Symbol']
	df_cnv = df_cnv.drop(df_cnv.columns[0], axis=1) # remving tcga ids from mRNA data
	df_cnv = df_cnv.apply(pd.to_numeric) # convert all columns of DataFrame to numeric
	high_variance_genes = df_cnv.var().nlargest(500) # selecting top 500 genes based on variance
	# print(high_variance_genes)
	# print(high_variance_genes.index)
	# print(df_cnv[high_variance_genes.index])
	df_cnv = df_cnv[high_variance_genes.index] #selecting 500 genes from the dataframe
	# df_mRNA.insert(0, "submitter_id.samples", survival_id.values, True)
	df_cnv.insert(loc = len(df_cnv.columns),column = 'label_cnv',value = class_labels.values)
	df_cnv.insert(loc = 0,column = 'submitter_id.samples',value = tcga_id.values)	# re-inserting the submitter_id.samples
	# print(df_cnv)
	return df_cnv

def dna_raw_feature():
	print('Reading a large file...ETA 10 min')
	dna_file_name_27 = 'HumanMethylation27'
	dna_file_name_450 = 'HumanMethylation450'

	df_dna_27 = pd.read_csv(os.path.join(path,'dna_methylation',dna_file_name_27), header=None,index_col=0, delimiter="\t", low_memory=False).T# read the csv data file and transpose it
	df_dna_450 = pd.read_csv(os.path.join(path,'dna_methylation',dna_file_name_450), header=None,index_col=0, delimiter="\t", low_memory=False).T# read the csv data file and transpose it
	# print(df_dna_27)
	# print(df_dna_450)
	df_dna_column = df_dna_450.columns.intersection(df_dna_27.columns) # identifying common features (columns) between methylation 27 and 450
	df_dna_27 = df_dna_27[df_dna_column] #selecting samples based on common features
	df_dna_450 = df_dna_450[df_dna_column] #selecting samples based on common features
	df_dna = df_dna_450.append(df_dna_27)
	df_dna.drop_duplicates(subset ="sample",keep='first',inplace=True)
	df_dna=df_dna.sort_values(by=['sample'],ascending=True) # sorting based on tcga ids
	df_dna.drop_duplicates(subset ="sample",keep='first',inplace=True)
	df_dna_id=df_dna['sample'] # extracting tcga ids from dna dataframe
	# df_dna_id = df_dna_id.str[:-1]
	# df_dna = df_dna.drop('sample', axis=1) # remving old tcga ids from data
	# df_dna.insert(0, "sample", df_dna_id.values, True) #adding the modified tcga_id
	df_survival_label = pd.read_csv(os.path.join(path,'survival_class','5_year_survival.csv'),delimiter=',') # reading survival csv file
	survival_df = df_survival_label[df_survival_label['submitter_id.samples'].isin(df_dna_id)] # selcting survival label of tcga ids matching with dna tcga ids
	survival_df=survival_df.sort_values(by=['submitter_id.samples'],ascending=True) # sorting survival labels based on tcga ids
	survival_df.drop_duplicates(subset ="submitter_id.samples",keep='first',inplace=True)
	survival_id = survival_df['submitter_id.samples'] # extracting tcga ids from survival labels
	df_dna = df_dna[df_dna['sample'].isin(survival_id)] # selecting dna data of patients matching with avivalable survival idsclass_labels = survival_df['5_year_cutoff'] # fetching the class labels
	df_dna.drop_duplicates(subset ="sample",keep='first',inplace=True)
	tcga_id = df_dna['sample']
	class_labels = survival_df['5_year_cutoff'] # fetching the class labels
	df_dna.dropna(how='any', axis=1, inplace=True) # droping column with all missing values

	df_dna = df_dna.drop('sample', axis=1)
	df_dna = df_dna.apply(pd.to_numeric) # convert all columns of DataFrame to numeric
	high_variance_dna = df_dna.var().nlargest(500) # selecting top 500 genes based on variance
	# print(high_variance_dna)
	# print(high_variance_dna.index)
	# print(df_dna[high_variance_dna.index])
	df_dna = df_dna[high_variance_dna.index] #selecting 500 genes from the dataframe
	min_max_scaler = MinMaxScaler()

	df_dna[df_dna.columns] = min_max_scaler.fit_transform(df_dna[df_dna.columns])
	# from sklearn.impute import KNNImputer
	# imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
	# # fit on the dataset
	# print('missing values imputing')

	# X=df_dna.values
	# imputer.fit(X)
	# # transform the dataset
	# X = imputer.transform(X)
	# df_dna = pd.DataFrame(X, columns = df_dna.columns)
	df_dna.insert(loc = len(df_dna.columns),column = 'label_dna',value = class_labels.values)
	df_dna.insert(loc = 0,column = 'submitter_id.samples',value = tcga_id.values)	# re-inserting the submitter_id.samples
	print(df_dna)
	return df_dna

def mir_raw_feature():
	miR_file_name = 'TCGA-BRCA.mirna.tsv'
	df_miR = pd.read_csv(os.path.join(path,'miRSeq',miR_file_name), header=None,index_col=0, delimiter="\t", low_memory=False).T# read the csv data file and transpose it
	df_miR=df_miR.sort_values(by=['miRNA_ID'],ascending=True) # sorting based on tcga ids
	
	# print(df_miR)
	df_miR_id=df_miR['miRNA_ID'] # extracting tcga ids from miR dataframe
	df_miR_id = df_miR_id.str[:-1]
	df_miR = df_miR.drop('miRNA_ID', axis=1) # remving old tcga ids from data
	df_miR.insert(0, "miRNA_ID", df_miR_id.values, True) #adding the modified tcga_id
	df_survival_label = pd.read_csv(os.path.join(path,'survival_class','5_year_survival.csv'),delimiter=',') # reading survival csv file
	survival_df = df_survival_label[df_survival_label['submitter_id.samples'].isin(df_miR_id)] # selcting survival label of tcga ids matching with miR tcga ids
	survival_df = survival_df.sort_values(by=['submitter_id.samples'],ascending=True) # sorting survival labels based on tcga ids
	survival_df.drop_duplicates(subset ="submitter_id.samples",keep='first',inplace=True)
	survival_id = survival_df['submitter_id.samples'] # extracting tcga ids from survival labels
	df_miR = df_miR[df_miR['miRNA_ID'].isin(survival_id)] # selecting miR data of patients matching with avivalable survival ids
	df_miR.drop_duplicates(subset ='miRNA_ID',keep='first',inplace=True)
	tcga_id = df_miR['miRNA_ID']
	class_labels = survival_df['5_year_cutoff'] # fetching the class labels
	df_miR = df_miR.drop('miRNA_ID', axis=1) # remving old tcga ids from data
	df_miR.dropna(how='any', axis=1, inplace=True) # droping column with all missing values
	# from sklearn.impute import KNNImputer
	# imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
	# # fit on the dataset
	# print('missing values imputing')

	# X=df_miR.values
	# imputer.fit(X)
	# # transform the dataset
	# X = imputer.transform(X)
	# df_miR = pd.DataFrame(X, columns = df_miR.columns)
	df_miR = df_miR.apply(pd.to_numeric) # convert all columns of DataFrame to numeric
	high_variance_mir = df_miR.var().nlargest(500) # selecting top 500 genes based on variance
	# print(high_variance_dna)
	# print(high_variance_dna.index)
	# print(df_dna[high_variance_dna.index])
	df_miR = df_miR[high_variance_mir.index] #selecting 500 genes from the dataframe
	
	min_max_scaler = MinMaxScaler()
	df_miR[df_miR.columns] = min_max_scaler.fit_transform(df_miR[df_miR.columns])
	df_miR.insert(loc = len(df_miR.columns),column = 'label_mir',value = class_labels.values)
	df_miR.insert(loc = 0,column = 'submitter_id.samples',value = tcga_id.values)	# re-inserting the submitter_id.samples

	return df_miR


# def mir_raw_feature():
# 	miRNA_GA_gene_file_name = 'miRNA_GA_gene'
# 	miRNA_HiSeq_gene_file_name = 'miRNA_HiSeq_gene'
# 	df_miR_GA = pd.read_csv(os.path.join(path,'miRSeq',miRNA_GA_gene_file_name), header=None,index_col=0, delimiter="\t", low_memory=False).T# read the csv data file and transpose it
# 	df_miR_HiSeq = pd.read_csv(os.path.join(path,'miRSeq',miRNA_HiSeq_gene_file_name), header=None,index_col=0, delimiter="\t", low_memory=False).T# read the csv data file and transpose it
# 	df_mir_columns = df_miR_HiSeq.columns.intersection(df_miR_GA.columns) # identifying common features (columns)
# 	df_miR_GA = df_miR_GA[df_mir_columns] #selecting samples based on common features
# 	df_miR_HiSeq = df_miR_HiSeq[df_mir_columns] #selecting samples based on common features
# 	df_miR = df_miR_HiSeq.append(df_miR_GA)
# 	df_miR.drop_duplicates(subset ="sample",keep='first',inplace=True)
# 	# print(df_miR)
# 	df_miR=df_miR.sort_values(by=['sample'],ascending=True) # sorting based on tcga ids
# 	df_miR_id=df_miR['sample'] # extracting tcga ids from miR dataframe
# 	df_survival_label = pd.read_csv(os.path.join(path,'survival_class','5_year_survival.csv'),delimiter=',') # reading survival csv file
# 	survival_df = df_survival_label[df_survival_label['submitter_id.samples'].isin(df_miR_id)] # selcting survival label of tcga ids matching with miR tcga ids
# 	survival_df = survival_df.sort_values(by=['submitter_id.samples'],ascending=True) # sorting survival labels based on tcga ids
# 	survival_df.drop_duplicates(subset ="submitter_id.samples",keep='first',inplace=True)
# 	survival_id = survival_df['submitter_id.samples'] # extracting tcga ids from survival labels
# 	df_miR = df_miR[df_miR['sample'].isin(survival_id)] # selecting miR data of patients matching with avivalable survival ids
# 	class_labels = survival_df['5_year_cutoff'] # fetching the class labels
# 	tcga_id = df_miR['sample']
# 	df_miR = df_miR.drop('sample', axis=1) # remving tcga ids from mRNA data
# 	df_miR.dropna(how='all', axis=1, inplace=True) # droping column with all missing values
# 	from sklearn.impute import KNNImputer
# 	imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
# 	# fit on the dataset
# 	print('missing values imputing')

# 	X=df_miR.values
# 	imputer.fit(X)
# 	# transform the dataset
# 	X = imputer.transform(X)
# 	df_miR = pd.DataFrame(X, columns = df_miR.columns)
# 	df_miR.insert(loc = len(df_miR.columns),column = 'label_mir',value = class_labels.values)
# 	df_miR.insert(loc = 0,column = 'submitter_id.samples',value = tcga_id.values)	# re-inserting the submitter_id.samples
# 	print(df_miR)
# 	return df_miR


def mrna_raw_feature():
	# mRNA_file_name = 'TCGA-BRCA.htseq_fpkm-uq_discrete.csv'
	# mRNA_file_name = 'TCGA-BRCA.htseq_fpkm-uq.tsv'
	mRNA_file_name = 'HiSeqV2_PANCAN'
	# transposing raw mRNA data and sorting them by patietnt Ids or TCGA Ids
	# =============================================================================
	df_mRNA = pd.read_csv(os.path.join(path,'mRNA',mRNA_file_name), header=None,index_col=0, delimiter="\t", low_memory=False).T# read the csv data file and transpose it
	df_mRNA.drop_duplicates(subset ="sample",keep='first',inplace=True)
	df_mRNA=df_mRNA.sort_values(by=['sample'],ascending=True) # sorting based on tcga ids
	print(df_mRNA)
	# =============================================================================
	#Mapping mRNA patient ids with survival labels
	# =============================================================================
	df_mRNA_id=df_mRNA['sample'] # extracting tcga ids from mRNA dataframe
	# df_mRNA_id = df_mRNA_id.str[:-1]
	# df_mRNA = df_mRNA.drop('Ensembl_ID', axis=1) # remving old tcga ids from data
	# df_mRNA.insert(0, "Ensembl_ID", df_mRNA_id.values, True) #adding the modified tcga_id
	df_survival_label = pd.read_csv(os.path.join(path,'survival_class','5_year_survival.csv'),delimiter=',') # reading survival csv file
	survival_df = df_survival_label[df_survival_label['submitter_id.samples'].isin(df_mRNA_id)] # selcting survival label of tcga ids matching with mRNA tcga ids
	survival_df=survival_df.sort_values(by=['submitter_id.samples'],ascending=True) # sorting survival labels based on tcga ids
	survival_df.drop_duplicates(subset ="submitter_id.samples",keep='first',inplace=True)
	survival_id = survival_df['submitter_id.samples'] # extracting tcga ids from survival labels
	df_mRNA = df_mRNA[df_mRNA['sample'].isin(survival_id)] # selecting mRNA data of patients matching with avivalable survival ids
	# print(df_mRNA)
	class_labels = survival_df['5_year_cutoff'] # fetching the class labels
	tcga_id = df_mRNA['sample']
	df_mRNA = df_mRNA.drop('sample', axis=1) # remving tcga ids from mRNA data
	# convert all columns of DataFrame
	# col =df_mRNA.columns.drop([df_mRNA.columns[0]])
	df_mRNA = df_mRNA.apply(pd.to_numeric) # convert all columns of DataFrame to numeric
	high_variance_genses = df_mRNA.var().nlargest(500) # selecting top 500 genes based on variance
	# print(high_variance_genses)
	# print(high_variance_genses.index)
	# print(df_mRNA[high_variance_genses.index])
	df_mRNA = df_mRNA[high_variance_genses.index] #selecting 500 genes from the dataframe
	min_max_scaler = MinMaxScaler()

	df_mRNA[df_mRNA.columns] = min_max_scaler.fit_transform(df_mRNA[df_mRNA.columns])
	
	df_mRNA.insert(loc = len(df_mRNA.columns),column = 'label_mrna',value = class_labels.values)
	df_mRNA.insert(loc = 0,column = 'submitter_id.samples',value = tcga_id.values)	# re-inserting the submitter_id.samples
	# print(df_mRNA)

	return df_mRNA

def wsi_raw_feature():
	wsi_file_name = 'tcga_allWSIpatchfused.csv'
	df_wsi = pd.read_csv(os.path.join(path,'wsi',wsi_file_name), header=0,index_col=None, delimiter=",", low_memory=False)# read the csv data file and transpose it
	print(df_wsi)
	
	return df_wsi


# cln_raw_feature()
# cnv_raw_feature()
# dna_raw_feature()
# mir_raw_feature()
# mrna_raw_feature()
# wsi_raw_feature()

cln_raw_feature().to_csv(os.path.join(path,'clinical','raw_features_cln_regression.csv'),index=False)
cnv_raw_feature().to_csv(os.path.join(path,'cnv','raw_features_cnv.csv'),index=False)
dna_raw_feature().to_csv(os.path.join(path,'dna_methylation','raw_features_dna.csv'),index=False)
mir_raw_feature().to_csv(os.path.join(path,'miRSeq','raw_features_mir.csv'),index=False)
mrna_raw_feature().to_csv(os.path.join(path,'mRNA','raw_features_mrna.csv'),index=False)
wsi_raw_feature().to_csv(os.path.join(path,'wsi','raw_features_wsi.csv'),index=False)