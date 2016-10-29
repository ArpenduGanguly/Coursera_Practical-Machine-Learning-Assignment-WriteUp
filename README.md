# Coursera_Practical-Machine-Learning-Assignment-WriteUp

“Practical Machine Learning Assignment WriteUp”
Author: Arpendu Kumar Ganguly date: “29 October 2016” output: html_document

“Project Overview”
This document was prepared as part of Project assignment for the course:Practical Machine Learning offered as a MOOC by John Hopkins University. The project entailed the conceptual understanding and practical application of various ML Techniques: Decision Trees, Random Forest, Gradient Boosting, Bagging, & cross-validation methods for predicting better out of sample accuracy.

I.Project Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks.

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

I.i) Project Introduction
In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participant They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

The five ways are exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). Only Class A corresponds to correct performance.

I.ii) Research Question
The goal of this project is to predict the manner in which they did the exercise, i.e., Class A to E. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

II. Preparing the Environment
The following packages will be used for the ML algorithms in the assignment.

library(knitr)
library(caret )
library(randomForest)
library(rattle)
library(rpart)
library(rpart.plot)
library(corrplot)
library(e1071)
library(forecast)
library(AppliedPredictiveModeling)
set.seed(323321)
III.Loading the data
The train set data is available at:

trainurl="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
The test set data is available at:

testurl="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
Importing the data to the local environment

pmltr=read.csv(url(trainurl))
pmlte=read.csv(url(testurl))
dim(pmltr)
## [1] 19622   160
dim(pmlte)
## [1]  20 160
We have 19,622 rows and 160 coloumns in train data and 20 rows and 160 coloumns in the test data

IV. Splitting the train data into training and testing sets
pmltresplit=createDataPartition(pmltr$classe, p=0.6 ,list = FALSE)
training=pmltr[pmltresplit,]
testing=pmltr[-pmltresplit,]
dim(training)
## [1] 11776   160
dim(testing)
## [1] 7846  160
V. Treatment of Data before model building
Since, there are 160 predictors, we check if we can reduce down the predictor size. We can do this, by elimating the predictors with Zero or Near Zero Variance, Identifying the near zero variance predictors

NZV1=nearZeroVar(training, saveMetrics = TRUE)
NZV=nearZeroVar(training)
Removing the near zero variance predictors set from the total set of predictors

training=training[,-NZV]
testing=testing[,-NZV]
dim(training)
## [1] 11776   107
dim(testing)
## [1] 7846  107
Treating the data for missing values: Further, a lot of predictors have missing values which is more than 95%. We remove them from the set of predictors

ALLNAs=sapply(training, function(x) mean(is.na(x)))>0.95
training=training[,ALLNAs==FALSE]
testing=testing[,ALLNAs==FALSE]
dim(training)  
## [1] 11776    59
dim(testing)
## [1] 7846   59
We also remove the ID Variables from the list of predictors

training=training[,-(1:5)]
testing = testing[,-(1:5)]
dim(training)  
## [1] 11776    54
dim(testing)
## [1] 7846   54
So, after the first level of data treatment, we have been able to scale down the list of predictors from 160 to 54 predictors.

VI. Expolatory Data Analysis
We plot a correlation matrix between the set of predictors, to see if strong relationships exists between them.



Since there, is no distinct correlation patterns coming up, we move to towards building Model Building. Here, we will try three model building methods, 1.Decision Trees 2. Random Forest 3. Gradient Boosting method

and choose the model based on best out-of-sample accuracy metric.

VII. Machine Learning Model Building
With Decision Trees
set.seed(323321)
modDectree=rpart(classe~.,data=training,method = "class")
fancyRpartPlot(modDectree, main= "Decision Tree on Classe")


modDectreet=predict(modDectree,testing, type ="class")
confusionMatrix(modDectreet,testing$classe)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1940  255   53   46   14
##          B   95  986  133  149  191
##          C   27   85 1104  202  120
##          D  134  192   78  826  172
##          E   36    0    0   63  945
## 
## Overall Statistics
##                                          
##                Accuracy : 0.7394         
##                  95% CI : (0.7295, 0.749)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.6701         
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8692   0.6495   0.8070   0.6423   0.6553
## Specificity            0.9344   0.9102   0.9330   0.9122   0.9845
## Pos Pred Value         0.8406   0.6345   0.7178   0.5892   0.9052
## Neg Pred Value         0.9473   0.9154   0.9581   0.9286   0.9269
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2473   0.1257   0.1407   0.1053   0.1204
## Detection Prevalence   0.2942   0.1981   0.1960   0.1787   0.1331
## Balanced Accuracy      0.9018   0.7799   0.8700   0.7772   0.8199
With Deicison Trees, we achieve a accuracy of 74.57%

With Random Forests
set.seed(323321)
modFitranfor=randomForest(classe~.,data=training)
predictrf=predict(modFitranfor,testing, type="class")
confusionMatrix(predictrf,testing$classe)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    2    0    0    0
##          B    0 1516   10    0    0
##          C    0    0 1357    6    0
##          D    0    0    1 1280    6
##          E    0    0    0    0 1436
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9968          
##                  95% CI : (0.9953, 0.9979)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.996           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9987   0.9920   0.9953   0.9958
## Specificity            0.9996   0.9984   0.9991   0.9989   1.0000
## Pos Pred Value         0.9991   0.9934   0.9956   0.9946   1.0000
## Neg Pred Value         1.0000   0.9997   0.9983   0.9991   0.9991
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1932   0.1730   0.1631   0.1830
## Detection Prevalence   0.2847   0.1945   0.1737   0.1640   0.1830
## Balanced Accuracy      0.9998   0.9986   0.9955   0.9971   0.9979
With Random Forest, we achieve a accuracy of 99.49

To check, which predictors are most important for predicting the classe variable

varImpPlot(modFitranfor)


With Genralized Boosting Method
set.seed(323321)
fitControl=trainControl( method ="repeatedcv", number = 5, repeats = 1)

gbmfit1=train(classe~.,data= training, method ="gbm", trControl=fitControl, verbose=FALSE)

gbmpred=predict(gbmfit1,newdata=testing)
gbmaccuracytest=confusionMatrix(gbmpred,testing$classe)
gbmaccuracytest
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2229   11    0    1    0
##          B    2 1491   10    6    8
##          C    0   11 1356    9    1
##          D    1    5    2 1269   16
##          E    0    0    0    1 1417
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9893          
##                  95% CI : (0.9868, 0.9915)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9865          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9987   0.9822   0.9912   0.9868   0.9827
## Specificity            0.9979   0.9959   0.9968   0.9963   0.9998
## Pos Pred Value         0.9946   0.9829   0.9847   0.9814   0.9993
## Neg Pred Value         0.9995   0.9957   0.9981   0.9974   0.9961
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2841   0.1900   0.1728   0.1617   0.1806
## Detection Prevalence   0.2856   0.1933   0.1755   0.1648   0.1807
## Balanced Accuracy      0.9983   0.9891   0.9940   0.9916   0.9913
With Genralized Boosting Method, we achieve a accuracy of 98.36

Hence, we have the accuracy for each model: Decision Tree:74.57 Random Forest:99.49 Genralized Boosting Method:98.36

Thus, we select Random Forest for prediction in the test data set

VIII. Prediction in the test data set
finalpredict=predict(modFitranfor, pmlte, type = "class" )
finalpredict
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
“Preparing the files for final submission”

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
