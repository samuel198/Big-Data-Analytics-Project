#!/usr/bin/env python
# coding: utf-8

# APACHE SPARK PROJECT CODES USING LOGISTIC REGRESSION AND RANDOM FOREST MODELS.

# In[64]:


import os
import sys
 
os.environ["SPARK_HOME"] = "/usr/hdp/current/spark2-client"
os.environ["PYLIB"] = os.environ["SPARK_HOME"] + "/python/lib"
# In below two lines, use /usr/bin/python2.7 if you want to use Python 2
os.environ["PYSPARK_PYTHON"] = "/usr/local/anaconda/bin/python" 
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/anaconda/bin/python"
sys.path.insert(0, os.environ["PYLIB"] +"/py4j-0.10.4-src.zip")
sys.path.insert(0, os.environ["PYLIB"] +"/pyspark.zip")


# In[65]:


from pyspark.sql import SparkSession
spark=SparkSession.builder.getOrCreate()


# In[66]:


df = spark.read.csv('weather/seattleweather.csv',
                       header=True,inferSchema=True)


# In[67]:


df.columns


# In[68]:


df.printSchema()


# In[69]:


df.take(6)


# In[70]:


for col in df.columns:
    print("no. of cells in column", col, "with null values:",
          df.filter(df[col].isNull()).count())


# In[71]:


df.count()


# In[72]:


df.describe(['RAIN']).show()


# In[73]:


#Label encoder
from pyspark.ml.feature import StringIndexer
indexed = df
for col in ["RAIN"]:
    stringIndexer = StringIndexer(inputCol=col, outputCol=col+"_encoded")
    model = stringIndexer.fit(indexed)
    indexed = model.transform(indexed)
indexed.show(3)


# In[74]:


df = indexed.select(indexed.DATE,indexed.PRCP,indexed.TMAX,indexed.TMIN,indexed.RAIN_encoded)
df.show(3)


# In[75]:


# Split the data into train and test sets
featurecols = df.columns[2:]

from pyspark.ml.feature import VectorAssembler, StandardScaler
assembler = VectorAssembler(inputCols=featurecols, 
                            outputCol="features")
df_feature_vec=assembler.transform(df)


# In[76]:


# Split the data into train and test sets
train_data, test_data = df_feature_vec.randomSplit([.8,.2],seed=1200)

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=True)
scalerModel = scaler.fit(train_data)
scaledData = scalerModel.transform(train_data)
scaledData_test = scalerModel.transform(test_data)
scaledData.select("RAIN_encoded","scaledFeatures").take(3) 


# LOGISTIC REGRESSION

# In[77]:


from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.mllib.util import MLUtils
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# In[78]:


# Create ParamGrid for Cross Validation
from pyspark.ml.classification import LogisticRegression
evaluator=BinaryClassificationEvaluator(rawPredictionCol="scaledFeatures",labelCol="RAIN_encoded")
lr = LogisticRegression(labelCol="RAIN_encoded", featuresCol="scaledFeatures",maxIter=10)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
paramGrid = ParamGridBuilder()    .addGrid(lr.aggregationDepth,[2,5,10])    .addGrid(lr.elasticNetParam,[0.0, 0.5, 1.0])    .addGrid(lr.fitIntercept,[False, True])    .addGrid(lr.maxIter,[10, 100, 1000])    .addGrid(lr.regParam,[0.01, 0.5, 2.0])     .build()


# In[79]:


# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
# Run cross validations
cvModel = cv.fit(scaledData)
predict_train=cvModel.transform(scaledData)
predict_test=cvModel.transform(scaledData_test)
print("The area under ROC for train set after CV  is {}".format(evaluator.evaluate(predict_train)))
print("The area under ROC for test set after CV  is {}".format(evaluator.evaluate(predict_test)))


# In[80]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator=BinaryClassificationEvaluator(rawPredictionCol="scaledFeatures",                                        labelCol="RAIN_encoded")
predict_test.select("RAIN_encoded","rawPrediction","prediction",                    "probability").show(5)


# In[81]:


evaluator = MulticlassClassificationEvaluator(
        labelCol="RAIN_encoded", predictionCol="prediction", metricName="accuracy")
accuracy=evaluator.evaluate(predict_test)
print(accuracy)
evaluator = MulticlassClassificationEvaluator(
        labelCol="RAIN_encoded", predictionCol="prediction", metricName="weightedRecall")
recall=evaluator.evaluate(predict_test)
print(recall)
evaluator = MulticlassClassificationEvaluator(
        labelCol="RAIN_encoded", predictionCol="prediction", metricName="weightedPrecision")
precision=evaluator.evaluate(predict_test)
print(precision)


# In[82]:


predict_test.select(['RAIN_encoded','prediction', 'probability']).show(5)


# In[83]:


predict_test.select('RAIN_encoded','prediction').groupby('RAIN_encoded','prediction').count().sort('prediction').sort('RAIN_encoded').show()


# RANDOM FOREST

# In[84]:


# Create ParamGrid for Cross Validation
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

rf = RandomForestClassifier(labelCol="RAIN_encoded", featuresCol="scaledFeatures")

paramGrid = (ParamGridBuilder()             .addGrid(rf.maxDepth, [2, 6])             .addGrid(rf.numTrees, [5, 20])             .build())

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

cvModel = cv.fit(scaledData)


# In[85]:


# Use test set here so we can measure the accuracy of our model on new data
predictions_rf = cvModel.transform(scaledData_test)
evaluator.evaluate(predictions_rf)


# In[86]:


evaluator = MulticlassClassificationEvaluator(
        labelCol="RAIN_encoded", predictionCol="prediction", metricName="accuracy")
accuracy=evaluator.evaluate(predictions_rf)
print(accuracy)
evaluator = MulticlassClassificationEvaluator(
        labelCol="RAIN_encoded", predictionCol="prediction", metricName="weightedRecall")
recall=evaluator.evaluate(predictions_rf)
print(recall)
evaluator = MulticlassClassificationEvaluator(
        labelCol="RAIN_encoded", predictionCol="prediction", metricName="weightedPrecision")
precision=evaluator.evaluate(predictions_rf)
print(precision)


# In[87]:


predictions_rf.select(['RAIN_encoded','prediction', 'probability']).show(5)


# In[88]:


predictions_rf.select('RAIN_encoded','prediction').groupby('RAIN_encoded','prediction').count().sort('prediction').sort('RAIN_encoded').show()


# In[89]:


spark.stop()

