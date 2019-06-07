#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import string
from nltk.corpus import stopwords
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.utils import np_utils
from numpy import array
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers import LeakyReLU
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Input


df_test_Main = pd.read_csv('./AppDetails_Reviews_Combined.csv',index=False)
print(df_test_Main.shape)
df_test_Main.info()

#############################################################################################
#Rescaling and Normalizing - Installs, Total Average Rating and Total Number of Reviews, Required Android Version
#############################################################################################
from sklearn.preprocessing import MinMaxScaler
print('------------------------------Installs----------------------------------------')
df_test_Main['Installs_1']=(df_test_Main['Installs']/1000000000)
print(df_test_Main['Installs_1'].unique())
print('-------------------------------Tot Avg Rate-----------------------------------')
df_test_Main['TotalAverageRating_1']=(df_test_Main['TotalAverageRating']/5)
print(df_test_Main['TotalAverageRating_1'].unique())
print('------------------------------------------------------------------------------')

TotNoRev_main=np.array(df_test_Main.TotNumRev)
cs = MinMaxScaler()
TotNoRev_main_1=TotNoRev_main.reshape(-1, 1)
print('------------------------------Tot No of Reviews-------------------------------')
TotNoRev_main_2 = cs.fit_transform(TotNoRev_main_1)
df_test_Main['TotNumRev2']=TotNoRev_main_2
print(df_test_Main['TotNumRev2'].unique())
print('------------------------------------------------------------------------------')

required_android_version_1=np.array(df_test_Main.required_android_version)
cs2 = MinMaxScaler()
required_android_version_2=required_android_version_1.reshape(-1, 1)
print('------------------------------Required android version-------------------------------')
required_android_version_2 = cs2.fit_transform(required_android_version_2)
df_test_Main['required_android_version_2']=required_android_version_2
print(df_test_Main['required_android_version_2'].unique())
print('------------------------------------------------------------------------------')

#############################################################################################
#Total: 63340 
#X: Main_Category_UniqueVal,TotNumRev,TotalAverageRating_1,required_android_version, Installs_1
#Y: current_rating
#############################################################################################

X = df_test_Main.drop(labels = ['current_rating','rev_app_id','TotalNumOfReviews','Installs','CleanedStop_review_body1','TotalAverageRating','TotNumRev','required_android_version'],axis = 1)

y = df_test_Main.current_rating
print('------------------------------------X value - Input----------------------------------------')
print(X.head(5))
print('-------------------------------------------------------------------------------------')

print('------------------------------------y value - output----------------------------------')
#print(y)
print('-------------------------------------------------------------------------------------')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
print('Split Done')
n_cols = X_train.shape[1]
n_cols0 = X_train.shape[0]

print('-------------------------------------------------------------------------------------')
print('Entire shape of X_train :::: ',X_train.shape)
print('-------------------------------------------------------------------------------------')
shapey=y_train.shape
print('-------------------------------------------------------------------------------------')
print('Entire shape of y_train :::: ',shapey)
print('-------------------------------------------------------------------------------------')

Y_rating_ip = np.array(y_train)
Y_rating_op = np.array(y_test)

Y_rating_bin_ip = np_utils.to_categorical(Y_rating_ip)
print('-----------------------------------Y_rating_bin_ip  shape-----------------------------------')
print(Y_rating_bin_ip.shape)
print('-------------------------------------------------------------------------------------')
Y_rating_bin_op = np_utils.to_categorical(Y_rating_op)
print('-----------------------------------Y_rating_bin_op shape------------------------------------')
print(Y_rating_bin_op.shape)
print('-------------------------------------------------------------------------------------')
#Controlling Epochs and Batch Size value throughout
epochs_val=25
batch_size_val=400

#############################################################################################
# create MLP model for App details
#############################################################################################

#Basic MLP
model_det_RevRating = Sequential()
model_det_RevRating.add(Dense(30,activation='relu',input_shape=(n_cols,)))
model_det_RevRating.add(Dense(18, activation='relu'))
model_det_RevRating.add(LeakyReLU(alpha=0.1))
model_det_RevRating.add(Dense(6, activation='softmax'))
print(model_det_RevRating.summary())


#Deep Dumb MLP (DDMLP)
nb_classes=6
model_det_RevRating_DD = Sequential()
model_det_RevRating_DD.add(Dense(24,input_dim=n_cols))
model_det_RevRating_DD.add(Activation('relu'))
model_det_RevRating_DD.add(Dropout(0.4))
model_det_RevRating_DD.add(Dense(18))
model_det_RevRating_DD.add(LeakyReLU(alpha=0.1))
model_det_RevRating_DD.add(Dropout(0.2))
model_det_RevRating_DD.add(Dense(nb_classes))
model_det_RevRating_DD.add(Activation('softmax'))

print("-----------------------------------------------------------------------------------------------------")
print(model_det_RevRating_DD.summary())
#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=50)


model_det_RevRating.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#model_det_RevRating.fit(X_train, Y_rating_bin_ip,validation_data=(X_test, Y_rating_bin_op),epochs=50, callbacks=[early_stopping_monitor],batch_size=20)
model_det_RevRating.fit(X_train, Y_rating_bin_ip,validation_data=(X_test, Y_rating_bin_op),epochs=epochs_val,batch_size=batch_size_val)
print("-----------------------------------------------------------------------------------------------------")
model_det_RevRating_DD.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#model_det_RevRating_DD.fit(X_train, Y_rating_bin_ip,validation_data=(X_test, Y_rating_bin_op),epochs=50, callbacks=[early_stopping_monitor],batch_size=20)
model_det_RevRating_DD.fit(X_train, Y_rating_bin_ip,validation_data=(X_test, Y_rating_bin_op),epochs=epochs_val,batch_size=batch_size_val)



print("---------------------- Basic MLP -----------------------")
loss_train_mlp, accuracy_train_mlp = model_det_RevRating.evaluate(X_train, Y_rating_bin_ip, verbose=0)
print("Training Accuracy: {:.4f}".format(accuracy_train_mlp))
loss_test_mlp, accuracy_test_mlp  = model_det_RevRating.evaluate(X_test, Y_rating_bin_op, verbose=0)
print("Testing Accuracy:  {:.4f}".format(accuracy_test_mlp))
preds_mlp = model_det_RevRating.predict_classes(X_test, verbose=0)
print("Predicted MLP::")
print(preds_mlp)

print("---------------------- DDMLP ----------------------------")
loss_train_ddmlp, accuracy_train_ddmlp = model_det_RevRating_DD.evaluate(X_train, Y_rating_bin_ip, verbose=0)
print("Training Accuracy: {:.4f}".format(accuracy_train_ddmlp))
loss_test_ddmlp, accuracy_test_ddmlp = model_det_RevRating_DD.evaluate(X_test, Y_rating_bin_op, verbose=0)
print("Testing Accuracy:  {:.4f}".format(accuracy_test_ddmlp))
preds_ddmlp = model_det_RevRating_DD.predict_classes(X_test, verbose=0)
print("Predicted DDMLP::")
print(preds_ddmlp)


#############################################################################################
# CONVERTING THE Text Reviews INTO A VECTOR using Count Vectorizer and TF IDF Vectorizer
#############################################################################################

from sklearn.feature_extraction.text import CountVectorizer
import string
from nltk.corpus import stopwords
from sklearn import preprocessing
from keras.utils import to_categorical

# CLASSIFICATION
data_classes = df_test_Main[(df_test_Main['current_rating']==1) | (df_test_Main['current_rating']==2) | (df_test_Main['current_rating']==3) | (df_test_Main['current_rating']==4) | (df_test_Main['current_rating']==5)]
data_classes.head()
#print('fetched shape of data classes:',data_classes.shape)
data_classes1=data_classes.dropna()

# Seperate the dataset into X and Y for classification 
X_Text = data_classes1['CleanedStop_review_body1']
Y_Rate = data_classes1['current_rating']
Z_Cat = data_classes1['Main_Category_UniqueVal']
print(X_Text.shape)
print(Y_Rate.shape)
#print(Z_Cat.shape)

#Using TfIdf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=18000)
tfidf_vect.fit(X_Text)
X_Main_vec_tf=tfidf_vect.transform(X_Text)

#Split into train set and test set 
RevText_train, RevText_test, RevRate_train, RevRate_test,Cat_train,Cat_test,tf_train,tf_test = train_test_split(X_Text, Y_Rate,Z_Cat,X_Main_vec_tf, test_size=0.30)
print('Split Done')

Y_RevRate_ip = to_categorical(RevRate_train)
print('---------------------------------User Rating Shape Train-----------------------------')
print('Y_RevRate_ip-- shape of 1 DIM::: ',Y_RevRate_ip.shape[1])
print('Y_RevRate_ip-- One Elem::: ',Y_RevRate_ip[0])
print('-------------------------------------------------------------------------------------')
Y_RevRate_op = to_categorical(RevRate_test)
print('---------------------------------User Rating Shape Test------------------------------')
print('Y_RevRate_op-- shape of 1 DIM::: ',Y_RevRate_op.shape[1])
print('Y_RevRate_op-- One Elem::: ',Y_RevRate_op[0])
print('-------------------------------------------------------------------------------------')

#Using Count Vectorizer
vectorizer = CountVectorizer(min_df=0.01)
vectorizer.fit(RevText_train)
X_train_vec = vectorizer.transform(RevText_train)
X_test_vec  = vectorizer.transform(RevText_test) 
print('X_train_vec shape::')
print(X_train_vec.shape)
print('X_train_vec-- 1 element ::')
print(X_train_vec[0])
print('X_test_vec shape ::')
print(X_test_vec.shape)
print('X_test_vec --1 element ::')
print(X_test_vec[0])
print('---------------------------------No of Cols for Count Vectorizer---------------------')
n_cols_Vec = X_train_vec.shape[1]
print('n_cols_Vec-- 1 Dim of Input X_train_vec:::: ',n_cols_Vec)
print('---------------------------------No of Cols for TF IDF Vectorizer--------------------')
n_cols_Vec_tf=tf_train.shape[1]
print('n_cols_Vec_tf::',n_cols_Vec_tf)





#############################################################################################
# create MLP model for App Reviews 
#model_Reviews_rate - For Text vectorized using Count Vectorizer 
#model_Reviews_rate_tf - For Text vectorized using TFIDF Vectorizer 
#############################################################################################
# create model
model_Reviews_rate = Sequential()
model_Reviews_rate.add(Dense(25,activation='relu',input_shape=(n_cols_Vec,)))
model_Reviews_rate.add(Dense(15, activation='relu'))
model_Reviews_rate.add(LeakyReLU(alpha=0.1))
model_Reviews_rate.add(Dense(6, activation='softmax'))
early_stopping_monitor = EarlyStopping(patience=25)
print(model_Reviews_rate.summary())

model_Reviews_rate_tf = Sequential()
model_Reviews_rate_tf.add(Dense(25,activation='relu',input_shape=(n_cols_Vec_tf,)))
model_Reviews_rate_tf.add(Dense(15, activation='relu'))
model_Reviews_rate_tf.add(LeakyReLU(alpha=0.1))
model_Reviews_rate_tf.add(Dense(6, activation='softmax'))
early_stopping_monitor = EarlyStopping(patience=25)
print(model_Reviews_rate_tf.summary())

model_Reviews_rate.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#history = model_Reviews_rate.fit(X_train_vec, Y_RevRate_ip,validation_data=(X_test_vec, Y_RevRate_op),epochs=50,callbacks=[early_stopping_monitor],batch_size=20)
history1 = model_Reviews_rate.fit(X_train_vec, Y_RevRate_ip,validation_data=(X_test_vec, Y_RevRate_op),epochs=epochs_val,batch_size=batch_size_val)

model_Reviews_rate_tf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history2 = model_Reviews_rate_tf.fit(tf_train, Y_RevRate_ip,validation_data=(tf_test, Y_RevRate_op),epochs=epochs_val,batch_size=batch_size_val)


loss_train_revrate, accuracy_train_revrate = model_Reviews_rate.evaluate(X_train_vec, Y_RevRate_ip, verbose=0)
print("Training Accuracy: {:.4f}".format(accuracy_train_revrate))
loss_test_revrate, accuracy_test_revrate = model_Reviews_rate.evaluate(X_test_vec, Y_RevRate_op, verbose=0)
print("Testing Accuracy:  {:.4f}".format(accuracy_test_revrate))
Preds_Mlp_ReviewRate = model_Reviews_rate.predict_classes(X_test_vec, verbose=0)
print("Predicted Preds_Mlp_ReviewRate::")
print(Preds_Mlp_ReviewRate)
print('-----------------------------------------------------------------------------------')
loss_train_revrate_tf, accuracy_train_revrate_tf = model_Reviews_rate_tf.evaluate(tf_train, Y_RevRate_ip, verbose=0)
print("Training Accuracy: {:.4f}".format(accuracy_train_revrate_tf))
loss_test_revrate_tf, accuracy_test_revrate_tf = model_Reviews_rate_tf.evaluate(tf_test, Y_RevRate_op, verbose=0)
print("Testing Accuracy:  {:.4f}".format(accuracy_test_revrate_tf))
Preds_Mlp_ReviewRate_tf = model_Reviews_rate_tf.predict_classes(tf_test, verbose=0)
print("Predicted Preds_Mlp_ReviewRate::")
print(Preds_Mlp_ReviewRate_tf)
print('------------------------------------------------------------------------------------')


#############################################################################################
# create Functional Combined MLP model - (App Details + Reviews ) #Using TF IDF Vectorizer
#############################################################################################

#Using TFIDF Vectorizer 
#First Input Model
visible11 = Input(shape=(n_cols,))
hidden11 = Dense(25, activation='relu')(visible11)
hidden21 = Dense(18, activation='relu')(hidden11)
hidden31 = Dense(15, activation='relu')(hidden21)
output11 = Dense(6, activation='softmax')(hidden31)
#flat1 = Flatten()(output)

# second input model
visible22 = Input(shape=(n_cols_Vec_tf,))
hidden42 = Dense(25,activation='relu')(visible22)
hidden52 = Dense(15,activation='relu')(hidden42)
output22 = Dense(6, activation='softmax')(hidden52)
           
# merge input models
merge12 = concatenate([output11, output22])
           
hidden63 = Dense(25, activation='relu')(merge12)
hidden73 = Dense(15, activation='relu')(hidden63)
output33 = Dense(6, activation='softmax')(hidden73)
model_combo12 = Model(inputs=[visible11, visible22], outputs=output33)
# summarize layers
print(model_combo12.summary())

model_combo12.compile(optimizer='rmsprop' , loss = "sparse_categorical_crossentropy", metrics=["accuracy"])
Test_output_param_tfidf=np.array(RevRate_train)
#early_stopping_monitor = EarlyStopping(patience=10)

Test_output_param=np.array(RevRate_test)
history3=model_combo12.fit([X_train,tf_train] ,kaaam_chalao,validation_data=([X_test, tf_test],Test_output_param),epochs=epochs_val,batch_size=batch_size_val)

#Test_output_param=np.array(RevRate_test)

loss_train_combo12, accuracy_train_combo12 = model_combo12.evaluate([X_train,tf_train] ,Test_output_param_tfidf, verbose=0)
print("Training Accuracy: {:.4f}".format(accuracy_train_combo12))
loss_test_combo12, accuracy_test_combo12 = model_combo12.evaluate([X_test,tf_test] ,Test_output_param, verbose=0)
print("Testing Accuracy:  {:.4f}".format(accuracy_test_combo12))


#############################################################################################
# create Functional Combined MLP model - (App Details + Reviews ) - #Using COunt Vectorizer
#############################################################################################

from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Input

#First Input Model
visible1 = Input(shape=(n_cols,))
hidden1 = Dense(25, activation='relu')(visible1)
hidden2 = Dense(18, activation='relu')(hidden1)
hidden3 = Dense(15, activation='relu')(hidden2)
output1 = Dense(6, activation='softmax')(hidden3)
#flat1 = Flatten()(output)

# second input model
visible2 = Input(shape=(n_cols_Vec,))
hidden4 = Dense(25,activation='relu')(visible2)
hidden5 = Dense(15,activation='relu')(hidden4)
output2 = Dense(6, activation='softmax')(hidden5)
           
# merge input models
merge = concatenate([output1, output2])
           
hidden6 = Dense(25, activation='relu')(merge)
hidden7 = Dense(15, activation='relu')(hidden6)
output3 = Dense(6, activation='softmax')(hidden7)
model_combo = Model(inputs=[visible1, visible2], outputs=output3)
# summarize layers
print(model_combo.summary())


model_combo.compile(optimizer='rmsprop' , loss = "sparse_categorical_crossentropy", metrics=["accuracy"])
kaaam_chalao=np.array(RevRate_train)

history4=model_combo.fit([X_train,X_train_vec] ,kaaam_chalao,validation_data=([X_test, X_test_vec],Test_output_param), epochs=epochs_val,batch_size=batch_size_val)

Test_output_param=np.array(RevRate_test)
loss_train_combo, accuracy_train_combo = model_combo.evaluate([X_train,X_train_vec] ,Test_output_param_tfidf, verbose=0)
print("Training Accuracy: {:.4f}".format(accuracy_train_combo))
loss_test_combo, accuracy_test_combo = model_combo.evaluate([X_test,X_test_vec] ,Test_output_param, verbose=0)
print("Testing Accuracy:  {:.4f}".format(accuracy_test_combo))
 
#############################################################################################
# Plot Train Vs Test Accuracy & Loss for Functional Combined MLP model - (App Details + Reviews ) 
# For Count Vectorizer and Using TF IDF Vectorizer
#############################################################################################


import matplotlib.pyplot as plt1

history=history4
print('For TF IDF Vectorized Content - Combined model::')
loss_1 = history.history['loss']
val_loss= history.history['val_loss']
epochs_1 = range(1, len(loss_1) + 1)

plt1.plot(epochs_1,loss_1,'bo',label='Traning loss')
plt1.plot(epochs_1,val_loss,'b',label='Validation loss')
plt1.title('Training and Validation loss')
plt1.xlabel('Epochs')
plt1.ylabel('Loss')

plt1.legend()
plt1.show()

print('For Count Vectorized Content - Combined model::')
history=history3
loss_1 = history.history['loss']
val_loss= history.history['val_loss']
epochs_1 = range(1, len(loss_1) + 1)

plt1.plot(epochs_1,loss_1,'bo',label='Traning loss')
plt1.plot(epochs_1,val_loss,'b',label='Validation loss')
plt1.title('Training and Validation loss')
plt1.xlabel('Epochs')
plt1.ylabel('Loss')
plt1.legend()
plt1.show()


import matplotlib.pyplot as plt2
history=history4
print('For TF IDF Vectorized Content - Combined model')

acc_1 = history.history['acc']
val_acc= history.history['val_acc']
#epochs_1 = range(1, len(loss) + 1)

plt2.plot(epochs_1,acc_1,'bo',label='Training acc')
plt2.plot(epochs_1,val_acc,'b',label='Validation acc')
plt2.title('Training and Validation accuracy')
plt2.xlabel('Epochs')
plt2.ylabel('Loss')
plt2.legend()

plt2.show()

history=history3
print('For Count Vectorized Content - Combined model')

acc_1 = history.history['acc']
val_acc= history.history['val_acc']
#epochs_1 = range(1, len(loss) + 1)

plt2.plot(epochs_1,acc_1,'bo',label='Training acc')
plt2.plot(epochs_1,val_acc,'b',label='Validation acc')
plt2.title('Training and Validation accuracy')
plt2.xlabel('Epochs')
plt2.ylabel('Accuracy')
plt2.legend()

