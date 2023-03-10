import pandas as pd 
import os
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import layers


cancer_type =  ['BRCA']
index = 0
os.environ["CUDA_VISIBLE_DEVICES"]="1"
path = '/Data/nikhilanand_1921cs24/VAE_SVM/'+cancer_type[index]

df_cln = pd.read_csv(os.path.join(path,'clinical','raw_features_cln.csv'), header=0,index_col=None, delimiter=",",low_memory=False)# read the csv data file

tcga_id = df_cln[df_cln.columns[0]]
class_labels = df_cln[df_cln.columns[len(df_cln.columns)-1]]
# =============================================================================
df_cln = df_cln.drop(df_cln.columns[0], axis=1) # droping tcga id from first column
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
df_cln[["age_at_initial_pathologic_diagnosis"]] = min_max_scaler.fit_transform(df_cln[["age_at_initial_pathologic_diagnosis"]])


X = df_cln.drop(df_cln.columns[len(df_cln.columns)-1], axis=1).values # dropping class label from last column
y = class_labels.values # storing class label in separate variable




from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca = PCA(n_components=0.95)
X_transformed=pca.fit_transform(X)
columns =[]
for i in range(1,X_transformed.shape[1]+1):
    temp = 'cln_pca'+str(i)
    columns.append(temp)

df_cln_pca = pd.DataFrame(X_transformed, columns = columns)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('Explained variance')
plt.savefig('elbow_plot.png',dpi=100)
df_cln_pca.insert(0, "submitter_id.samples", tcga_id.values, True)
df_cln_pca.insert(loc = len(df_cln_pca.columns),column = 'label_cln',value = class_labels.values)
output_file_name = 'pca_features_cln.csv'
df_cln_pca.to_csv(os.path.join(path,'clinical',output_file_name),index=False)
print("Files have been saved to: "+os.path.join(path))




from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

original_dim = X.shape[1]
intermediate_dim = 8
latent_dim = 4

LOSS_THRESHOLD = 0.01

class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('loss') < LOSS_THRESHOLD):
			print("\nReached %2.2f%% loss, so stopping training!!" %(LOSS_THRESHOLD*100))
			self.model.stop_training = True

# Instantiate a callback object
callbacks = myCallback()


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Define encoder model.
original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
x = layers.Dense(intermediate_dim, activation="tanh")(original_inputs)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()((z_mean, z_log_var))
encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name="encoder")

# Define decoder model.
latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
x = layers.Dense(intermediate_dim, activation="tanh")(latent_inputs)
outputs = layers.Dense(original_dim)(x)
decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name="decoder")

# Define VAE model.
outputs = decoder(z)
vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="vae")

# Add KL divergence regularization loss.
kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
vae.add_loss(kl_loss)

# Train.
import tensorflow_addons as tfa
initial_learning_rate = 1e-3
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer = tfa.optimizers.Lookahead(optimizer)
vae.compile(optimizer, loss=tf.keras.losses.LogCosh())
vae.fit(X,X, epochs=100, batch_size =32,callbacks=[callbacks])
latent_features =  encoder.predict(X)
columns =[]
for i in range(1,latent_features.shape[1]+1):
    temp = 'cln_vae_'+str(i)
    columns.append(temp)
latent_df = pd.DataFrame(latent_features,columns = columns)
latent_df.insert(loc = 0,column = 'submitter_id.samples',value = tcga_id.values)
latent_df.insert(loc = len(latent_df.columns),column = 'label_cln',value = class_labels.values)
latent_df.to_csv(os.path.join(path,'clinical','vae_features_cln.csv'),index=False,header=True)


