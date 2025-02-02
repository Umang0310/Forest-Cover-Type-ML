import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df=pd.read_csv('covtype.csv')
df_without_soil = df.drop(df.columns[10:54], axis=1)

features = df_without_soil.drop('Cover_Type', axis=1)
target = df_without_soil['Cover_Type']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

