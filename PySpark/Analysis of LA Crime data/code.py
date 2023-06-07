#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
findspark.init()


# In[2]:


import pandas as pd


# In[3]:


import pyspark
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()
print(sc.version)
print(spark.version)


# In[ ]:


conf = pyspark.SparkConf().set("spark.jars.packages", 
                              "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1").setMaster("local").setAppName("App").setAll([("spark.driver.memory","40g"),("spark.executor.memory","50g")])


# In[ ]:


sc = SparkContext(conf = conf)


# In[ ]:


sqlC = SQLContext(sc)


# In[ ]:


mongo_ip = "mongodb://localhost:27017/ece552Group18."


# In[ ]:


la_crime_df = sqlC.read.format("com.mongodb.spark.sql.DefaultSource").option("uri", mongo_ip + "la_crime").load()


# In[ ]:


la_crime_df.createOrReplaceTempView("la_crime_df")


# In[4]:


get_ipython().run_cell_magic('time', '', '\ndirPath = ""\nfileName = "LA_Crime.csv"')


# In[5]:


la_crime_df = spark.read.format("csv").                 option("header", "true").option("mode", "DROPMALFORMED").option("delimiter",",").             option("ignoreLeadingWhiteSpace","true").option("ignoreTrailingWhiteSpace","true"). option("inferschema","true").load(dirPath + fileName)


# In[6]:


la_crime_df.show(5)


# In[7]:


la_crime_df.printSchema()


# In[8]:


la_crime_df.count()


# In[6]:


la_crime_df = la_crime_df.drop("DR_NO","Part 1-2","Mocodes","Vict Descent",
                               "Status","Crm Cd 1","Crm Cd 2",
                               "Crm Cd 3","Crm Cd 4","LOCATION","Cross Street",
                               "LAT","LON") 
                 
la_crime_df.printSchema()


# In[7]:


la_crime_df = la_crime_df.withColumnRenamed("Date_Rptd","Date_Reported") .withColumnRenamed("DATE OCC","Date_Occurred") .withColumnRenamed("TIME OCC","Time_Occurred") .withColumnRenamed("AREA NAME","Area_Name") .withColumnRenamed("Rpt Dist No","Reported_District_No") .withColumnRenamed("Crm Cd","Crime_Code") .withColumnRenamed("Crm Cd Desc","Crime_Code_Description") .withColumnRenamed("Vict Age","Victim_Age") .withColumnRenamed("Vict Sex","Victim_Sex") .withColumnRenamed("Premis Cd","Premises_Code") .withColumnRenamed("Premis Desc","Premises_Description") .withColumnRenamed("Weapon Used Cd","Weapon_Used_Code") .withColumnRenamed("Weapon Desc","Weapon_Description") .withColumnRenamed("Status Desc","Status_Description")

la_crime_df.printSchema()


# In[8]:


la_crime_df.write.parquet("C:/BigData/~notebookJupyter/crime.parquet")


# In[9]:


la_crime_parquet = spark.read.parquet("C:/BigData/~notebookJupyter/crime.parquet")


# In[13]:


la_crime_parquet.printSchema()


# In[10]:


la_crime_parquet = la_crime_parquet.dropna()

la_crime_parquet.count()


# In[15]:


from pyspark.sql.functions import desc


# In[16]:


#•	What demographics do majority of the victims fall under?

age = la_crime_parquet.groupBy("Victim_Age").count()
age.sort(desc("count")).show()

sex = la_crime_parquet.groupBy("Victim_Sex").count()
sex.sort(desc("count")).show()


# In[17]:


sex_pandas = sex.toPandas()

explode = (0.1,0.1,0.1,0.1)
sex_pandas.groupby(['Victim_Sex']).sum().plot(kind='pie', y='count', autopct='%1.0f%%',
                                colors = ['Palegreen','Lavender','Lightcoral','Skyblue'],
                                explode = explode,
                                shadow = True,
                                figsize=(10, 7),
                                textprops={'fontsize': 14},
                                startangle = 90,
                                title='Proportion of Genders of Victims')


# In[18]:


#•	What type of crime was committed more frequently?

crime = la_crime_parquet.groupBy("Crime_Code", "Crime_Code_Description").count()

crime.sort(desc("count")).show()


# In[29]:


import matplotlib.pyplot as plt

crime_pandas = crime.toPandas()

crime_pandas_sorted = crime_pandas.sort_values('count')
crime_pandas_sorted.plot.barh(x='Crime_Code_Description', y='count',
              title='Number of crimes per type', color='darkorchid', figsize=(15, 40))
plt.yticks(fontsize = 15)


# In[19]:


#•	What premises have more crimes been committed in?

premise = la_crime_parquet.groupBy("Premises_Description").count()
premise.sort(desc("count")).show()

area = la_crime_parquet.groupBy("Area_Name").count()
area.sort(desc("count")).show()


# In[27]:


area_pandas = area.toPandas()

area_pandas.plot.bar(x='Area_Name', y='count',
              title='Number of crimes per Area', color='teal', figsize=(15, 10))
plt.xticks(rotation=45, fontsize = 15)
plt.yticks(fontsize = 12)


# In[20]:


#•	What weapon has been used the most to commit the crimes?

weapon = la_crime_parquet.groupBy("Weapon_Description").count()
weapon.sort(desc("count")).show()


# In[26]:


weapon_pandas = weapon.toPandas()

# Plot a bar chart

weapon_pandas_sorted = weapon_pandas.sort_values('count')
weapon_pandas_sorted.plot.barh(x='Weapon_Description', y='count',
              title='Count of Weapons used', color='green', figsize=(15, 30))
plt.yticks(fontsize = 12)


# In[21]:


#•	What is the rate of prosecution for each of the crime types?

arrest = la_crime_parquet.groupBy("Status_Description").count()
arrest.sort(desc("count")).show()


# In[22]:


arrest_pandas = arrest.toPandas()

explode = (0.1,0.1,0.1,0.1,0.1)
arrest_pandas.groupby(['Status_Description']).sum().plot(kind='pie', y='count', autopct='%1.0f%%',
                                colors = ['Salmon','Turquoise','Plum','Cornflowerblue','HotPink'],
                                explode = explode,
                                shadow = True,
                                figsize=(10,10),
                                textprops={'fontsize': 14},
                                title='Proportion of Current Statuses of Cases')


# In[23]:


v_age = [378, 502, 2161, 8184, 12048, 13986, 13605, 9961, 7904, 8399, 6221, 5061, 3621, 2191, 1235, 674]
index = ['5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80'] 
df = pd.DataFrame({'Victim_Age': v_age},
                   index = index)
ax = df.plot.bar(rot=0, figsize=(10, 7))


# In[24]:


count = [50543, 49592, 51651, 50957, 53710, 54342, 55404, 55207, 52832, 54673, 46394, 33287]
index = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'] 
df1 = pd.DataFrame({'Number of Crimes': count},
                   index = index)

df1.plot.line(figsize=(10, 7))


# In[11]:


# String Indexing

from pyspark.ml.feature import StringIndexer

Area_Name_indexer = StringIndexer(inputCol="Area_Name", outputCol="Area_Name_index")
la_crime_parquet = Area_Name_indexer.fit(la_crime_parquet).transform(la_crime_parquet)

Date_Occurred_indexer = StringIndexer(inputCol="Date_Occurred", outputCol="Date_Occurred_index")
la_crime_parquet = Date_Occurred_indexer.fit(la_crime_parquet).transform(la_crime_parquet)

Date_Reported_indexer = StringIndexer(inputCol="Date_Reported", outputCol="Date_Reported_index")
la_crime_parquet = Date_Reported_indexer.fit(la_crime_parquet).transform(la_crime_parquet)

Crime_Code_Description_indexer = StringIndexer(inputCol="Crime_Code_Description", outputCol="Crime_Code_Description_index")
la_crime_parquet = Crime_Code_Description_indexer.fit(la_crime_parquet).transform(la_crime_parquet)

Victim_Sex_indexer = StringIndexer(inputCol="Victim_Sex", outputCol="Victim_Sex_index")
la_crime_parquet = Victim_Sex_indexer.fit(la_crime_parquet).transform(la_crime_parquet)

Premises_Description_indexer = StringIndexer(inputCol="Premises_Description", outputCol="Premises_Description_index")
la_crime_parquet = Premises_Description_indexer.fit(la_crime_parquet).transform(la_crime_parquet)

Weapon_Description_indexer = StringIndexer(inputCol="Weapon_Description", outputCol="Weapon_Description_index")
la_crime_parquet = Weapon_Description_indexer.fit(la_crime_parquet).transform(la_crime_parquet)

Status_Description_indexer = StringIndexer(inputCol="Status_Description", outputCol="Status_Description_index")
la_crime_parquet = Status_Description_indexer.fit(la_crime_parquet).transform(la_crime_parquet)


# In[31]:


la_crime_parquet.printSchema()


# In[12]:


ML_columns = ['Time_Occurred', 'AREA', 'Reported_District_No', 'Crime_Code', 'Victim_Age', 'Premises_Code', 'Weapon_Used_Code', 'Area_Name_index', 'Date_Occurred_index', 'Date_Reported_index', 'Crime_Code_Description_index', 'Premises_Description_index', 'Weapon_Description_index', 'Status_Description_index']


# In[13]:


#Vector Assembler

from pyspark.ml.feature import VectorAssembler
vector_assembler = VectorAssembler(inputCols = ML_columns, outputCol = 'features')


# In[15]:


assembled_df = vector_assembler.transform(la_crime_parquet)

assembled_df.select("features").show(10)


# In[35]:


train, test = assembled_df.randomSplit([0.7,0.3])


# In[36]:


train.count()


# In[37]:


test.count()


# In[38]:


###LOGISTIC REGRESSION - VICTIM GENDER

from pyspark.ml.classification import LogisticRegression


# In[39]:


LR = LogisticRegression(featuresCol = 'features', labelCol = 'Victim_Sex_index')
LR_model = LR.fit(train)

LR_pred = LR_model.transform(test)

LR_pred.select('prediction', 'Victim_Sex_index').show()


# In[40]:


#### Model Evaluation

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

LR_evaluator = MulticlassClassificationEvaluator(labelCol = 'Victim_Sex_index', metricName = 'accuracy')
LR_evaluator.evaluate(LR_pred)


# In[41]:


from pyspark.mllib.evaluation import MulticlassMetrics


# In[42]:


LR_metrics = MulticlassMetrics(LR_pred['Victim_Sex_index','prediction'].rdd)

print("Accuracy:", LR_metrics.accuracy)
print("Recall", LR_metrics.recall(1.0))
print("Fmeasure:", LR_metrics.fMeasure(1.0))
print("Precision:", LR_metrics.precision(1.0))
print("True Positive Rate", LR_metrics.truePositiveRate(1.0))


# In[43]:


###LOGISTIC REGRESSION - CRIME CODE

LR_crmcode = LogisticRegression(featuresCol = 'features', labelCol = 'Crime_Code_Description_index')
LR_model_crmcode = LR_crmcode.fit(train)

LR_pred_crmcode = LR_model_crmcode.transform(test)
LR_pred_crmcode.select('prediction', 'Crime_Code_Description_index').show()


# In[44]:


#### Model Evaluation

LR_evaluator_crmcode = MulticlassClassificationEvaluator(labelCol = 'Crime_Code_Description_index', metricName = 'accuracy')
LR_evaluator_crmcode.evaluate(LR_pred_crmcode)


# In[45]:


LR_metrics_crmcd = MulticlassMetrics(LR_pred_crmcode['Crime_Code_Description_index','prediction'].rdd)

print("Accuracy:", LR_metrics_crmcd.accuracy)
print("Recall", LR_metrics_crmcd.recall(1.0))
print("Fmeasure:", LR_metrics_crmcd.fMeasure(1.0))
print("Precision:", LR_metrics_crmcd.precision(1.0))
print("True Positive Rate", LR_metrics_crmcd.truePositiveRate(1.0))


# In[46]:


###LOGISTIC REGRESSION - CASE STATUS

LR_status = LogisticRegression(featuresCol = 'features', labelCol = 'Status_Description_index')
LR_model_status = LR_status.fit(train)

LR_pred_status = LR_model_status.transform(test)
LR_pred_status.select('prediction', 'Status_Description_index').show()


# In[47]:


LR_evaluator_status = MulticlassClassificationEvaluator(labelCol = 'Status_Description_index', metricName = 'accuracy')
LR_evaluator_status.evaluate(LR_pred_status)


# In[48]:


LR_metrics_status = MulticlassMetrics(LR_pred_status['Status_Description_index','prediction'].rdd)

print("Accuracy:", LR_metrics_status.accuracy)
print("Recall", LR_metrics_status.recall(1.0))
print("Fmeasure:", LR_metrics_status.fMeasure(1.0))
print("Precision:", LR_metrics_status.precision(1.0))
print("True Positive Rate", LR_metrics_status.truePositiveRate(1.0))


# In[49]:


###LOGISTIC REGRESSION - WEAPON

LR_weapon = LogisticRegression(featuresCol = 'features', labelCol = 'Weapon_Description_index')
LR_model_weapon = LR_weapon.fit(train)

LR_pred_weapon = LR_model_weapon.transform(test)
LR_pred_weapon.select('prediction', 'Weapon_Description_index').show()


# In[50]:


LR_evaluator_weapon = MulticlassClassificationEvaluator(labelCol = 'Weapon_Description_index', metricName = 'accuracy')
LR_evaluator_weapon.evaluate(LR_pred_weapon)


# In[51]:


LR_metrics_weapon = MulticlassMetrics(LR_pred_weapon['Weapon_Description_index','prediction'].rdd)

print("Accuracy:", LR_metrics_weapon.accuracy)
print("Recall", LR_metrics_weapon.recall(1.0))
print("Fmeasure:", LR_metrics_weapon.fMeasure(1.0))
print("Precision:", LR_metrics_weapon.precision(1.0))
print("True Positive Rate", LR_metrics_weapon.truePositiveRate(1.0))


# In[52]:


###LOGISTIC REGRESSION - PREMISES

LR_premises = LogisticRegression(featuresCol = 'features', labelCol = 'Premises_Description_index')
LR_model_premises = LR_premises.fit(train)

LR_pred_premises = LR_model_premises.transform(test)
LR_pred_premises.select('prediction', 'Premises_Description_index').show()


# In[53]:


#### Model Evaluation

LR_evaluator_premises = MulticlassClassificationEvaluator(labelCol = 'Premises_Description_index', metricName = 'accuracy')
LR_evaluator_premises.evaluate(LR_pred_premises)


# In[54]:


LR_metrics_premises = MulticlassMetrics(LR_pred_premises['Premises_Description_index','prediction'].rdd)

print("Accuracy:", LR_metrics_premises.accuracy)
print("Recall", LR_metrics_premises.recall(1.0))
print("Fmeasure:", LR_metrics_premises.fMeasure(1.0))
print("Precision:", LR_metrics_premises.precision(1.0))
print("True Positive Rate", LR_metrics_premises.truePositiveRate(1.0))


# In[55]:


###LOGISTIC REGRESSION - AREA NAME

LR_area = LogisticRegression(featuresCol = 'features', labelCol = 'Area_Name_index')
LR_model_area = LR_area.fit(train)

LR_pred_area = LR_model_area.transform(test)
LR_pred_area.select('prediction', 'Area_Name_index').show()


# In[56]:


LR_evaluator_area = MulticlassClassificationEvaluator(labelCol = 'Area_Name_index', metricName = 'accuracy')
LR_evaluator_area.evaluate(LR_pred_area)


# In[57]:


LR_metrics_area = MulticlassMetrics(LR_pred_area['Area_Name_index','prediction'].rdd)

print("Accuracy:", LR_metrics_area.accuracy)
print("Recall", LR_metrics_area.recall(1.0))
print("Fmeasure:", LR_metrics_area.fMeasure(1.0))
print("Precision:", LR_metrics_area.precision(1.0))
print("True Positive Rate", LR_metrics_area.truePositiveRate(1.0))


# In[59]:


##### MULTIVARIATE LINEAR REGRESSION #####

from pyspark.ml.regression import LinearRegression


# In[62]:


# Time Occurred

LReg_time = LinearRegression(featuresCol = 'features', labelCol = 'Time_Occurred')
LReg_model_time = LReg_time.fit(train)

LReg_pred_time = LReg_model_time.transform(test)
LReg_pred_time.select('Time_Occurred', 'prediction', ).show()


# In[63]:


LReg_evaluator_time = MulticlassClassificationEvaluator(labelCol = 'Time_Occurred', metricName = 'accuracy')
LReg_evaluator_time.evaluate(LReg_pred_time)


# In[64]:


# VICTIM AGE

LReg_age = LinearRegression(featuresCol = 'features', labelCol = 'Victim_Age')
LReg_model_age = LReg_age.fit(train)

LReg_pred_age = LReg_model_age.transform(test)
LReg_pred_age.select('Victim_Age', 'prediction').show()


# In[65]:


LReg_evaluator_age = MulticlassClassificationEvaluator(labelCol = 'Victim_Age', metricName = 'accuracy')
LReg_evaluator_age.evaluate(LReg_pred_age)

