import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

data={
    'Age':[25,30,35,40,45,50,55,60,20,22],
    'Income':[40000,50000,60000,80000,100000,120000,140000,160000,20000,25000]
    'Bought':[0,0,1,1,1,1,1,1,0,0]
}
df=pd.DataFrame(data)

X=df[['Age','Income']]
y=df['Bought']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

if __name__ == '__main__':
    print('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
