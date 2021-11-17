import pickle_compat
pickle_compat.patch()
import pandas as pd
import numpy as np
import numpy as np
from numpy.fft import fft
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
import pickle
# from train import Extraction_of_features



def CGM_Chan(datafram):
    
    cgm_df_velocity = [0]
    for i in range(len(datafram)-1):
        cgm_df_velocity += [datafram[i+1]-datafram[i]]
    cgm_df_velocity.sort(reverse=True)
    return cgm_df_velocity[:2]

def RMS_Mov(datafram):
    CGM_RMS_mov = []
    for i in range(len(datafram)-5):
        Win_summation = sum([a * a for a in datafram[i:i+5]])
        Win_summation /= 5
        RMs = Win_summation**0.5
        CGM_RMS_mov.append(RMs)
    CGM_RMS_mov.sort(reverse=True)
    return CGM_RMS_mov[:2]

def get_mean(x):
    return np.array(np.split(x, 6)).mean(axis=1)[2:-1]

def get_std(x):
    return np.array(np.split(x, 6)).std(axis=1)[2:-1]

def CGM_FFT_compute(datafram):
    CGM_fft = np.abs(fft(datafram.values))
    CGM_fft = CGM_fft.tolist()
    CGM_fft.sort(reverse=True)

    return CGM_fft[:3]

def Extraction_of_features(Meal_dataframe):
    extracted_feature_vectors1 = []
    for i in range(len(Meal_dataframe)):
        extracted_feature_vectors1.append(CGM_Chan(Meal_dataframe.iloc[i]))

    extracted_feature_vectors2 = []
    for i in range(len(Meal_dataframe)):
        extracted_feature_vectors2.append(RMS_Mov(Meal_dataframe.iloc[i]))

    extracted_feature_vectors3 = []
    for i in range(len(Meal_dataframe)):
        extracted_feature_vectors3.append(list(get_mean(Meal_dataframe.iloc[i])))

    extracted_feature_vectors4 = []
    for i in range(len(Meal_dataframe)):
        extracted_feature_vectors4.append(list(get_std(Meal_dataframe.iloc[i])))

    extracted_feature_vectors5 = []
    for i in range(len(Meal_dataframe)):
        extracted_feature_vectors5.append(CGM_FFT_compute(Meal_dataframe.iloc[i]))

    feature1 =  pd.DataFrame(extracted_feature_vectors1, columns = ['cms_v1', 'cms_v2' ])
    feature2 =  pd.DataFrame(extracted_feature_vectors2,  columns = ['rms1', 'rms2'])
    feature3 = pd.DataFrame(extracted_feature_vectors3,  columns = ['m1', 'm2', 'm3' ])
    feature4 = pd.DataFrame(extracted_feature_vectors5,  columns = ['s1', 's2','s3' ])
    feature5 = pd.DataFrame(extracted_feature_vectors5,  columns = ['ff1', 'ff2', 'ff3' ])
    MealData_Fin = pd.concat([feature1,feature2,feature3,feature4, feature5], axis=1)
    return MealData_Fin

test = pd.read_csv('test.csv',header=None)
Features_extracatedform_test=Extraction_of_features(test)

Priniciple_C_Analyssi = PCA(n_components=8)
Priniciple_C_Analyssi.fit_transform(Features_extracatedform_test)
X_Test = Priniciple_C_Analyssi.transform(Features_extracatedform_test)

with open('model.pkl', 'rb') as machine_trained_model:
    pickle_file = pickle.load(machine_trained_model)
    predict = pickle_file.predict(X_Test)    
    machine_trained_model.close()
pd.DataFrame(predict).to_csv('Results.csv',index=False,header=False)