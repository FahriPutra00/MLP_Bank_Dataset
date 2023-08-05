# library umum
import numpy as np 
import pandas as pd

# library sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

# library menampilkan data
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# VSC
dataset_bank = pd.read_csv('D:/CODING/SMT5/NNDL/DATASET/bank/bank-full.csv',sep=';')

dataset_bank=dataset_bank.drop_duplicates()
dataset_bank=dataset_bank.drop(['duration'], axis=1)
#Code here
num_cols = ['age','balance','day','campaign','pdays','previous']
for col in num_cols:
    # Mencari Quartile
    Q1 = dataset_bank[col].quantile(0.25)
    Q3 = dataset_bank[col].quantile(0.75)
    IQR = Q3 - Q1
    # Mencari Batas
    LB = Q1 - (IQR*1.5)
    UB = Q3 + (IQR*1.5)
    # Menghapus Outlier
    if col == 'campaign':
        dataset_bank = dataset_bank[(dataset_bank[col] < 38)]
    elif col == 'pdays':
        dataset_bank = dataset_bank[(dataset_bank[col] < 580)]
    elif col == 'previous':
        dataset_bank = dataset_bank[(dataset_bank[col] < 48)]
    else:
        dataset_bank = dataset_bank[(dataset_bank[col] > LB) & (dataset_bank[col] < UB)]
    
# encode strings to integer
dataset_bank['y'] = LabelEncoder().fit_transform(dataset_bank['y'])
dataset_bank['y']
dataset_bank['y'].value_counts()

education_mapper = {"unknown":-1, "primary":1, "secondary":2, "tertiary":3}
dataset_bank["education"] = dataset_bank["education"].replace(education_mapper)

# listing down the features that has categoricaldataset_bank
categorial_features = ['job', 'marital', 'contact', 'month', 'poutcome']
# categorial_features = ['job', 'marital', 'poutcome']
for item in categorial_features:
    # assigning the encodeddataset_bank into a new DataFrame object
    df = pd.get_dummies(dataset_bank[item], prefix=item)
    dataset_bank = dataset_bank.drop(item, axis=1)
    for categorial_feature in df.columns:
        #Set the new column indataset_bank to have corresponding df values
       dataset_bank[categorial_feature] = df[categorial_feature]
       
binary_valued_features = ['default','housing', 'loan']
bin_dict = {'yes':1, 'no':0}

#Replace binary values in dataset_bank using the provided dictionary
for item in binary_valued_features:
    dataset_bank.replace({item:bin_dict},inplace=True)

cols = list(dataset_bank.columns.values)
cols.pop(cols.index('y')) # pop y out of the list
dataset_bank = dataset_bank[cols+['y']] #Create new dataframe with columns in new 

y = dataset_bank['y']
X = dataset_bank.values[:, :-1] # get all columns except the last column
activation = ['identity', 'logistic', 'relu', 'tanh']
solver = ['adam', 'sgd', 'lbfgs']
hid_Layer = [(128,64,32,16,8), (128,64,32,16), (128,64,32), (128,64), 
             (128,64,32,16,8,4), (128,64,32,16,8,4,2),
             (128,64,32,16,8,4,2,1)]
learn_rate = [0.001, 0.01, 0.1]
act = []
solv = []
hd_lyr = []
lrt = []
acc = []

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Feature scaling
scaler = StandardScaler()  
scaler.fit(X)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)
  
correlation_matrix = pd.DataFrame(X_train).corr()
    
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
  
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape),k=1).astype(np.bool))

# checking which columns can be dropped
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
X_train = X_train.drop(X_train.columns[to_drop], axis=1)
X_test = X_test.drop(X_test.columns[to_drop], axis=1)
    
# apply the PCA for feature for feature reduction
pca = PCA(n_components=0.95)
pca.fit(X_train)
PCA_X_train = pca.transform(X_train)
PCA_X_test = pca.transform(X_test)
i=1
for actv in activation:
    for sol in solver:
        for hid in hid_Layer:
            for lr in learn_rate:
                mlp = MLPClassifier(hidden_layer_sizes=hid, max_iter=50, activation=actv, solver=sol, batch_size=32, learning_rate_init=lr, random_state=42, )
                hist = mlp.fit(PCA_X_train, y_train)
                act.append(actv)
                solv.append(sol)
                hd_lyr.append(hid)
                lrt.append(lr)
                # jd.append(j)
                acc.append(mlp.score(PCA_X_test, y_test))
                print(i)
                i+=1
        
print(f'Accuracy: {max(acc)} %')
print('index:', np.argmax(acc),
      '\nbest learning rate: ',lrt[np.argmax(acc)], 
      '\nBest hidden layer: ', hd_lyr[np.argmax(acc)],
      '\nBest MLP Activation: ', act[np.argmax(acc)],
      '\nBest MLP Solver: ', solv[np.argmax(acc)])


