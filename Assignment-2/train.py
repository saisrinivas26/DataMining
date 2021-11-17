from numpy.fft import fft
import pickle_compat
pickle_compat.patch() 
import pandas as pd
from sklearn import metrics
import pickle
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from datetime import timedelta as td


Raw_CGMData1=pd.read_csv('CGMData.csv', low_memory=False)
RawInsulinData1=pd.read_csv('InsulinData.csv', low_memory=False)
Raw_CGMData2=pd.read_csv('CGM_patient2.csv', low_memory=False)
RawInsulinData2=pd.read_csv('Insulin_patient2.csv', low_memory=False)
Raw_CGMData1['date_time']= Raw_CGMData1['Date'].str.cat(Raw_CGMData1['Time'],sep=" ")
Raw_CGMData1['date_time']=pd.to_datetime(Raw_CGMData1['date_time'])
Raw_CGMData2['date_time']= Raw_CGMData2['Date'].str.cat(Raw_CGMData2['Time'],sep=" ")
Raw_CGMData2['date_time']=pd.to_datetime(Raw_CGMData2['date_time'])

RawInsulinData1['date_time']= RawInsulinData1['Date'].str.cat(RawInsulinData1['Time'],sep=" ")
RawInsulinData1['date_time']=pd.to_datetime(RawInsulinData1['date_time'])

RawInsulinData2['date_time']= RawInsulinData2['Date'].str.cat(RawInsulinData2['Time'],sep=" ")
RawInsulinData2['date_time']=pd.to_datetime(RawInsulinData2['date_time'])



RawInsulinData1= RawInsulinData1[['BWZ Carb Input (grams)','date_time']]
RawInsulinData2= RawInsulinData2[['BWZ Carb Input (grams)','date_time']]

Raw_CGMData1=Raw_CGMData1[['Sensor Glucose (mg/dL)','date_time']]
Raw_CGMData2=Raw_CGMData2[['Sensor Glucose (mg/dL)','date_time']]


RawInsulinData1=RawInsulinData1.loc[(RawInsulinData1['BWZ Carb Input (grams)'].notnull()) & (RawInsulinData1['BWZ Carb Input (grams)']!=0) ]

RawInsulinData2=RawInsulinData2.loc[(RawInsulinData2['BWZ Carb Input (grams)'].notnull()) & (RawInsulinData2['BWZ Carb Input (grams)']!=0) ]


RawInsulinData1=RawInsulinData1.sort_values(['date_time']).reset_index(drop=True)
RawInsulinData2=RawInsulinData2.sort_values(['date_time']).reset_index(drop=True)




def Extract_MealData(InsulinData,CGMData):
    out=[]
    for i in range(len(InsulinData)):
        tm = (InsulinData.iloc[i]['date_time'])
        twohour = Find_twohoueMealData(InsulinData,tm,CGMData)
        if len(twohour)>0:
            result=(twohour['Sensor Glucose (mg/dL)'].tolist())
            if len(result)==30:
                out.append(result)
    Df_outer=pd.DataFrame(out)
    Df_outer=Df_outer.dropna()
    return Df_outer


def Find_twohoueMealData(InsulinData,Vl,CGMData):
    out_df=pd.DataFrame()
    Insulin_df=InsulinData[(InsulinData['date_time']>Vl) & (InsulinData['date_time']< Vl+ td(hours=2))]
    if len(Insulin_df)==0:
        if len(InsulinData[InsulinData['date_time']==Vl+td(hours=2)])>0:
            out_df=CGMData[(CGMData['date_time']>Vl+td(minutes=90)) & (CGMData['date_time']< Vl+ td(hours=4))]
        else:
            out_df=CGMData[(CGMData['date_time']>Vl-td(minutes=30)) & (CGMData['date_time']< Vl+ td(hours=2))]
    elif len(Insulin_df)>0:
        Vl=Insulin_df.iloc[0]['date_time']
        Find_twohoueMealData(InsulinData,Vl,CGMData)  
    return out_df
        
        
P1_Mealdata=Extract_MealData(RawInsulinData1,Raw_CGMData1)     
P2_MealData=Extract_MealData(RawInsulinData2,Raw_CGMData2)  


def Extract_NoMealData(Insulin,CGM):
    out=[]
    for i in range(len(Insulin)):
        tm = (Insulin.iloc[i]['date_time'])
        twohour_data=twohoutD(Insulin,tm,CGM)
        if len(twohour_data)>0:
            internal_list=(twohour_data['Sensor Glucose (mg/dL)'].tolist())
            if len(internal_list)==24:
                out.append(internal_list)
    result=pd.DataFrame(out)
    result=result.dropna()
    return result

def twohoutD(Insulin,Vl,CGMData):
    out = pd.DataFrame()
    InsulinD = Insulin[(Insulin['date_time']>Vl+td(hours=2)) & (Insulin['date_time']< Vl+ td(hours=4))]
    if len(InsulinD)==0:
        out = CGMData[(CGMData['date_time']>Vl+td(hours=2)) & (CGMData['date_time']< Vl+ td(hours=4))]
    return out
        
        
P1_NoMealData=Extract_NoMealData(RawInsulinData1,Raw_CGMData1)     
P2_NoMealData=Extract_NoMealData(RawInsulinData2,Raw_CGMData2)  

P_MealData=pd.concat([P1_Mealdata,P2_MealData])


P_NoMealData=pd.concat([P1_NoMealData,P2_NoMealData])



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

Feature_MealDF=Extraction_of_features(P_MealData)
Feature_NoMealDF=Extraction_of_features(P_NoMealData)


resultant1=[1]*len(Feature_MealDF)
resultant2=[0]*len(Feature_NoMealDF)

Feature_MealDF['resultant']=resultant1
Feature_NoMealDF['resultant']=resultant2


Combined_Data=pd.concat([Feature_MealDF,Feature_NoMealDF])


Combined_Data = Combined_Data.sample(frac = 1)


X=Combined_Data.iloc[:,:-1]
y=Combined_Data['resultant']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)


Priniciple_C_Analysis = PCA(n_components=8)
X_train = Priniciple_C_Analysis.fit_transform(X_train)
x_test = Priniciple_C_Analysis.transform(X_test)


svm_linearkernel = svm.SVC(kernel='linear')
svm_linearkernel.fit(X_train, y_train)
y_pred = svm_linearkernel.predict(x_test)

with open('model.pkl','wb') as f:
    pickle.dump(svm_linearkernel,f)





