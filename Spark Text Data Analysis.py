

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, count, regexp_replace, lower, isnan, udf, length, max as max_, row_number
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, HashingTF, IDF, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import IntegerType, BooleanType, DoubleType, ArrayType
from pyspark.ml.clustering import LDA
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import ClusteringEvaluator,MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors



from textblob import TextBlob
from langdetect import detect
import time





spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

#spark = SparkSession.builder.appName("SentimentAnalysis").config("spark.driver.memory", "120g").config("spark.executor.memory", "120g").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")




data = spark.read.csv("hashtag_joebiden.csv", header=True, inferSchema=True)
data.show(5)
print("\nTotal count of the data:", data.count())


data = data.dropna(how='any')
print("\nTotal count of the data after handling missing values", data.count())



# missingValues = data.agg(*[count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns])
# missingDict = missingValues.collect()[0].asDict()
# print(missingDict)

# ttlRows = data.count()
# missingPrcntg = {c: (v/ttlRows)*100 for c, v in missingDict.items()}
# print(missingPrcntg)




def sentiment_analysis(tweet):

    words = tweet.split()

    # Determine polarity
    polarity = TextBlob(" ".join(words)).sentiment.polarity

    # Classify overall sentiment
    if polarity > 0:
        sentiment = 1  # Positive
    elif polarity == 0:
        sentiment = 0  # Neutral
    else:
        sentiment = 2  # Negative

    return sentiment


# Create a UDF 
sentAnalysisUdf = udf(sentiment_analysis, IntegerType())
sentData = data.withColumn("sentiment", sentAnalysisUdf(data["tweet"]))
sentData.show(5)

# To get the count of each category
sentCounts = sentData.groupBy("sentiment").agg(count("*").alias("count"))
sentCounts.show()



def languageDetection(text):
    try:
        lang = detect(text)
        return lang == 'en'
    except:
        return False



langDetectorUdf = udf(languageDetection, BooleanType())
engData = sentData.filter(langDetectorUdf(sentData['tweet']))
engData.show(5)
print("\nTotal count of the data after filtering Non-English Tweets:", engData.count())




sentData = engData.withColumn("sentiment", when(col("sentiment") == 0, 2).otherwise(col("sentiment")))
sentData.show(5)

sentCounts = sentData.groupBy("sentiment").agg(count("*").alias("count"))
sentCounts.show(5)




sentimentData = sentData





cleanData = sentData.withColumn("tweet", regexp_replace(col("tweet"), "[^a-zA-Z\s]", ""))
cleanData = cleanData.withColumn("tweet", regexp_replace(col("tweet"), "\s+", " "))
cleanData = cleanData.withColumn("tweet", lower(col("tweet")))

tokenizer = Tokenizer(inputCol="tweet", outputCol="words")
tokenizeData = tokenizer.transform(cleanData) #cleanData

remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
filteredData = remover.transform(tokenizeData)

# vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="vec_features")
# vectorizerModel = vectorizer.fit(filteredData)
# vecData = vectorizerModel.transform(filteredData)

#Try using tf-Idf
#TFIDF
hashingTf = HashingTF(inputCol="filtered_words", outputCol="rawfeatures", numFeatures=40000) #numFeatures=10000
tfData = hashingTf.transform(filteredData)


idf = IDF(inputCol="rawfeatures", outputCol="features")
idfModel = idf.fit(tfData)
transformedData = idfModel.transform(tfData)
transformedData.show(5)




# Create and fit LDA model
lda = LDA(k=10, maxIter=10)
ldaModel = lda.fit(transformedData)

# Get the topics and their composition
topicsData = ldaModel.describeTopics(maxTermsPerTopic=10)
topicsData.show(truncate=False)





#to get the maximun probability and tweets assiciated with each topic

transformedDataWithTopics = ldaModel.transform(transformedData)

windowSpec = Window().orderBy(lda.getTopicDistributionCol())
transformedDataWithTopics = transformedDataWithTopics.withColumn("maxProbability", max_(lda.getTopicDistributionCol()).over(windowSpec))
filteredDataWithTopics = transformedDataWithTopics.filter(transformedDataWithTopics["maxProbability"] == transformedDataWithTopics[lda.getTopicDistributionCol()])
filteredDataWithTopics.select("topicDistribution", "tweet").show(5, truncate=False)


# ## Logistic Regression on data




# cleanData = sentimentData.withColumn("tweet", regexp_replace(col("tweet"), "[^a-zA-Z\s]", ""))
# cleanData = cleanData.withColumn("tweet", regexp_replace(col("tweet"), "\s+", " "))
# cleanData = cleanData.withColumn("tweet", lower(col("tweet")))





# tokenizer = Tokenizer(inputCol="tweet", outputCol="words")
# tokenizeData = tokenizer.transform(cleanData) #cleanData

# remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
# filteredData = remover.transform(tokenizeData)
# #filteredData.show(5)




# #TFIDF
# hashingTf = HashingTF(inputCol="filtered_words", outputCol="rawfeatures") #numFeatures=10000
# tfData = hashingTf.transform(filteredData)

# # Inverse Document Frequency
# idf = IDF(inputCol="rawfeatures", outputCol="features")
# idfModel = idf.fit(tfData)
# transformedData = idfModel.transform(tfData)
# transformedData.show(5)




#split data
trainData, testData = transformedData.randomSplit([0.7, 0.3], seed=42)


print(f"Number of training records: {trainData.count()}")
print(f"Number of test records: {testData.count()}")




#converting sentiments to 0 and 1, where 0=Positive and 1=Negative
trainData = trainData.withColumn("sentiment", when(trainData.sentiment == 2, 1).otherwise(0).cast("double"))
testData = testData.withColumn("sentiment", when(testData.sentiment == 2, 1).otherwise(0).cast("double"))





trainData.select("sentiment").show(5, truncate=False)
trainDataCounts = trainData.groupBy("sentiment").agg(count("*").alias("count"))
trainDataCounts.show(5)




lr = LogisticRegression(featuresCol="features", labelCol="sentiment")
lrModel = lr.fit(trainData)

trainPredictions = lrModel.transform(trainData)
testPredictions = lrModel.transform(testData)

trainRDD = trainPredictions.select("prediction", "sentiment").rdd
trainMetrics = MulticlassMetrics(trainRDD)

testRDD = testPredictions.select("prediction", "sentiment").rdd
testMetrics = MulticlassMetrics(testRDD)


trainAccuracy = trainMetrics.accuracy
testAccuracy = testMetrics.accuracy
precision = testMetrics.precision(1.0)
recall = testMetrics.recall(1.0)
f1Score = testMetrics.fMeasure(1.0)

print("\nLogistic Regression Summary Stats")
print(f"Logistic Regression Training Accuracy: {trainAccuracy*100:.2f}%")
print(f"Logistic Regression Test Accuracy:  {testAccuracy*100:.2f}%")
print("F1 Score = %s" % f1Score)





paramGridLR = (ParamGridBuilder()
               .addGrid(lr.regParam, [0.0, 0.5])
               .addGrid(lr.elasticNetParam, [0.0, 0.5])
               .addGrid(lr.maxIter, [5, 10])
               .build())


evaluator = MulticlassClassificationEvaluator(labelCol="sentiment", predictionCol="prediction", metricName="accuracy")
#evaluator = BinaryClassificationEvaluator(labelCol="sentiment", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

crossvalLR = CrossValidator(estimator=lr,
                            estimatorParamMaps=paramGridLR,
                            evaluator=evaluator,
                            numFolds=10)


lrTuneModel = crossvalLR.fit(trainData)
predictionsLR = lrTuneModel.transform(testData)

evaluatorLR = MulticlassClassificationEvaluator(labelCol="sentiment", predictionCol="prediction")
accuracyLR = evaluatorLR.evaluate(predictionsLR, {evaluator.metricName: "accuracy"})
f1LR = evaluatorLR.evaluate(predictionsLR, {evaluatorLR.metricName: "f1"})

bestModelParams = lrTuneModel.bestModel.extractParamMap()

regParamvalue = bestModelParams[lr.regParam]
elasticNetparam_value = bestModelParams[lr.elasticNetParam]
maxItervalue = bestModelParams[lr.maxIter]


print(f"\n ************************** \n")
print(f"\nLogistic Regression Best Accuracy and Parameters\n")
print(f"Logistic Regression - Best regParam: {regParamvalue}")
print(f"Logistic Regression - Best elasticNetParam: {elasticNetparam_value}")
print(f"Logistic Regression - Best maxIter: {maxItervalue}")
print(f"Accuracy: {accuracyLR*100:.2f}%, F1 Score: {f1LR}")




binaryEvaluator = BinaryClassificationEvaluator(labelCol="sentiment", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
roc_auc = binaryEvaluator.evaluate(predictionsLR)
print(f"Area Under ROC: {roc_auc}")






# ## On LDA Data with probability distribution




enhancedFeatureData = transformedDataWithTopics.select("tweet", "features", "topicDistribution", "sentiment")
enhancedFeatureData.show(5)





#to concatenate the features and topic distribution vectors
assembler = VectorAssembler(
    inputCols=["features", "topicDistribution"],
    outputCol="enhancedFeatures")

finalFeatureData = assembler.transform(enhancedFeatureData)





trainDataEnhanced, testDataEnhanced = finalFeatureData.randomSplit([0.7, 0.3], seed=42)





#converting sentiments to 0 and 1, where 0=Positive and 1=Negative
trainDataEnhanced = trainDataEnhanced.withColumn("sentiment", when(trainDataEnhanced.sentiment == 2, 1).otherwise(0).cast("double"))
testDataEnhanced = testDataEnhanced.withColumn("sentiment", when(testDataEnhanced.sentiment == 2, 1).otherwise(0).cast("double"))





trainDataEnhanced.select("sentiment").show(5, truncate=False)
trainDataEnhancedCounts = trainDataEnhanced.groupBy("sentiment").agg(count("*").alias("count"))
trainDataEnhancedCounts.show(5)





# Train the Logistic Regression model with enhanced features
lrEnhanced = LogisticRegression(featuresCol="enhancedFeatures", labelCol="sentiment")
lrModelEnhanced = lrEnhanced.fit(trainDataEnhanced)

testPredictionsEnhanced = lrModelEnhanced.transform(testDataEnhanced)

testMetricsEnhanced = MulticlassMetrics(testPredictionsEnhanced.select("prediction", "sentiment").rdd)


accuracyEnhanced = testMetricsEnhanced.accuracy
precisionEnhanced = testMetricsEnhanced.precision(1.0)
recallEnhanced = testMetricsEnhanced.recall(1.0)
f1ScoreEnhanced = testMetricsEnhanced.fMeasure(1.0)


print("\nEnhanced Logistic Regression Metrics")
print(f"Accuracy: {accuracyEnhanced*100:.2f}%")
print(f"Precision: {precisionEnhanced}")
print(f"Recall: {recallEnhanced}")
print(f"F1 Score: {f1ScoreEnhanced}")




# Compare the results
print("Improvement in Accuracy on Base Model: {:.2f}%".format((accuracyEnhanced - testAccuracy) * 100))
print("Improvement in F1 Score on Base Model: {:.2f}".format(f1ScoreEnhanced - f1Score))





#Tune LR Model
enhancedParamGridLR = (ParamGridBuilder()
               .addGrid(lrEnhanced.regParam, [0.0, 0.5])
               .addGrid(lrEnhanced.elasticNetParam, [0.0, 0.5])
               .addGrid(lrEnhanced.maxIter, [5, 10])
               .build())


enhancedEvaluator = MulticlassClassificationEvaluator(labelCol="sentiment", predictionCol="prediction", metricName="accuracy")
#evaluator = BinaryClassificationEvaluator(labelCol="sentiment", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

enhancedCrossvalLR = CrossValidator(estimator=lrEnhanced,
                            estimatorParamMaps=enhancedParamGridLR,
                            evaluator=enhancedEvaluator,
                            numFolds=10)


enhancedLRtuneModel = enhancedCrossvalLR.fit(trainDataEnhanced)
enhancedPredictionsLR = enhancedLRtuneModel.transform(testDataEnhanced)

enhancedEvaluatorLR = MulticlassClassificationEvaluator(labelCol="sentiment", predictionCol="prediction")
enhancedAccuracyLR = enhancedEvaluatorLR.evaluate(enhancedPredictionsLR, {enhancedEvaluator.metricName: "accuracy"})
enhancedF1LR = enhancedEvaluatorLR.evaluate(enhancedPredictionsLR, {enhancedEvaluatorLR.metricName: "f1"})

enhancedBestModelParams = enhancedLRtuneModel.bestModel.extractParamMap()

enhancedRegParamvalue = enhancedBestModelParams[lrEnhanced.regParam]
enhancedElasticNetparam_value = enhancedBestModelParams[lrEnhanced.elasticNetParam]
enhancedMaxItervalue = enhancedBestModelParams[lrEnhanced.maxIter]


print(f"\n ************************** \n")
print(f"\n Logistic Regression Best Accuracy and Parameters\n")
print(f"Logistic Regression - Best regParam: {enhancedRegParamvalue}")
print(f"Logistic Regression - Best elasticNetParam: {enhancedElasticNetparam_value}")
print(f"Logistic Regression - Best maxIter: {enhancedMaxItervalue}")
print(f"Accuracy: {enhancedAccuracyLR*100:.2f}%, F1 Score: {enhancedF1LR}")




# Compare the results
print("Improvement in Accuracy on Tuned Model: {:.2f}%".format((enhancedAccuracyLR - accuracyLR) * 100))
print("Improvement in F1 Score on Tuned Model: {:.2f}".format(enhancedF1LR - f1LR))







# # ## Graph Plotting 



# #ROC Graph
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc

# # Convert DataFrame to Pandas DataFrame
# # Make sure to collect only necessary data to avoid out of memory errors for large datasets
# predictionsDf = predictionsLR.select('sentiment', 'rawPrediction').toPandas()

# # Select the rawPrediction for the positive class
# predictionsDf['rawPrediction'] = predictionsDf['rawPrediction'].apply(lambda x: x[1])

# # Calculate ROC
# fpr, tpr, thresholds = roc_curve(predictionsDf['sentiment'], predictionsDf['rawPrediction'])
# roc_auc = auc(fpr, tpr)

# # Plot ROC curve
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()




# #Model comparision graph
# import matplotlib.pyplot as plt

# # Assuming you have calculated the improvement values for each metric
# # Replace these placeholders with the actual improvement values
# accuracy_improvements = [testAccuracy, accuracyEnhanced, 
#                         accuracyLR, enhancedAccuracyLR]

# f1_score_improvements = [f1Score, f1ScoreEnhanced, 
#                         f1LR, enhancedF1LR]

# # Model names
# models = ['Base Model', 'Enhanced Model', 'Tuned Model', 'Enhanced Tuned Model']

# # Plotting the bar graph for accuracy improvements
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.bar(models, accuracy_improvements, color='blue', alpha=0.7)
# plt.title('Improvement in Accuracy')
# plt.ylabel('Accuracy Improvement (%)')

# plt.xticks(rotation=90)

# # Plotting the bar graph for F1 score improvements
# plt.subplot(1, 2, 2)
# plt.bar(models, f1_score_improvements, color='green', alpha=0.7)
# plt.title('Improvement in F1 Score')
# plt.ylabel('F1 Score Improvement')

# plt.xticks(rotation=90)
# # Adjust layout to prevent overlapping
# plt.tight_layout()

# # Show the plot
# plt.show()




# import matplotlib.pyplot as plt
# from wordcloud import WordCloud, STOPWORDS
# from pyspark.ml.clustering import LDA
# import pyLDAvis
# import pyLDAvis.sklearn
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud, STOPWORDS
# from pyspark.ml.clustering import LDA
# import pyLDAvis
# import pyLDAvis.sklearn





# #sentimentData
# cleanData = sentimentData.withColumn("tweet", regexp_replace(col("tweet"), "[^a-zA-Z\s]", ""))
# cleanData = cleanData.withColumn("tweet", regexp_replace(col("tweet"), "\s+", " "))
# cleanData = cleanData.withColumn("tweet", lower(col("tweet")))

# # Create a WordCloud
# stopwords = set(STOPWORDS)
# wordcloud = WordCloud(
#     background_color='white',
#     stopwords=stopwords,
#     max_words=500,
#     max_font_size=40,
#     random_state=42
# ).generate(str(cleanData.select("tweet").rdd.flatMap(lambda x: x).collect()))

# print(wordcloud)
# fig = plt.figure(1)
# plt.imshow(wordcloud)
# plt.axis('off')
# plt.show()




# import gensim
# import pyLDAvis.gensim
# import gensim.corpora as corpora
# from gensim.models import LdaModel
# from gensim.models.coherencemodel import CoherenceModel
# import matplotlib.pyplot as plt


# data = cleanData.select('tweet').rdd.flatMap(lambda x: x).collect()

# tokenized_data = [gensim.utils.simple_preprocess(text) for text in data]
# dictionary = corpora.Dictionary(tokenized_data)
# corpus = [dictionary.doc2bow(text) for text in tokenized_data]




# # Replace these parameters with your desired values
# num_topics = 5
# lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)





# # Calculate the coherence score for different numbers of topics
# coherence_scores = []
# for num_topics in range(2, 11):
#     lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
#     coherence_model = CoherenceModel(model=lda_model, texts=tokenized_data, dictionary=dictionary, coherence='c_v')
#     coherence_score = coherence_model.get_coherence()
#     coherence_scores.append(coherence_score)

# # Plot the coherence scores to select the optimal number of topics
# plt.plot(range(2, 11), coherence_scores)
# plt.xlabel("Number of Topics")
# plt.ylabel("Coherence Score")
# plt.show()

# # Choose the number of topics with the highest coherence score
# optimal_num_topics = range(2, 11)[coherence_scores.index(max(coherence_scores))]


#pip instll numpy==1.25.2
#pip install pandas==1.5.3
#pip install pyLDAvis==3.4.0


# import gensim
# import pyLDAvis.gensim
# import gensim.corpora as corpora
# from gensim.models import LdaModel
# from gensim.models.coherencemodel import CoherenceModel
# import matplotlib.pyplot as plt


# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word, R=10)  # Adjust R as needed
# pyLDAvis.display(vis)







