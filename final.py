# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import isnull, when, count, col

# Ignoring warnings
warnings.filterwarnings("ignore")


# Load the dataset
df = pd.DataFrame(pd.read_csv("diabetes.csv"))

# Displaying dataset details
print(df.head())
print(df.describe())
print(df.info())
print(df.columns)
print(df.shape)
print(df.value_counts())
print(df.dtypes)

# Checking for missing values
print(df.isnull().sum())
print(df.isnull().any())

# Handling missing values by replacing 0 with NaN in specific columns
df_new = df.copy(deep=True)
df_new[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_new[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
print(df_new.isnull().sum())

# Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True)
plt.show()

# Histograms
df.hist(figsize=(10, 10))
plt.show()

# Pairplot for feature exploration
mean_col = ['Glucose', 'BloodPressure', 'Insulin', 'Age', 'Outcome', 'BMI']
sns.pairplot(df[mean_col], palette='Accent')
plt.show()

# Regression plot
sns.regplot(x='BMI', y='Glucose', data=df)
plt.show()

# XGBoost model training
# Assuming X_train, X_test, y_train, y_test are pre-defined
xgb_model = XGBClassifier(gamma=0)
xgb_model.fit(X_train, y_train)

# Model predictions and evaluation
xgb_pred = xgb_model.predict(X_test)
print("Accuracy Score =", metrics.accuracy_score(y_test, xgb_pred))
print(confusion_matrix(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))

# Feature importance plot
pd.Series(xgb_model.feature_importances_).plot(kind='barh')
plt.show()

# Spark Session initialization
spark = SparkSession.builder.appName('ml-diabetes').getOrCreate()
df_spark = spark.read.csv('diabetes.csv', header=True, inferSchema=True)
df_spark.printSchema()

# Display first 5 rows in Spark
print(pd.DataFrame(df_spark.take(5), columns=df_spark.columns).transpose())

# Count of rows based on outcome in Spark
print(df_spark.groupby('Outcome').count().toPandas())

# Checking for missing values in Spark
df_spark.select([count(when(isnull(c), c)).alias(c) for c in df_spark.columns]).show()

# Feature vectorization in Spark for machine learning
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
vector_assembler = VectorAssembler(inputCols=features, outputCol="features")
data_training_and_test = vector_assembler.transform(df_spark)
