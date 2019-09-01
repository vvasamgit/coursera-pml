
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library("gridExtra"))
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(randomForest))
suppressPackageStartupMessages(library(RColorBrewer))
suppressPackageStartupMessages(library(rattle))
suppressPackageStartupMessages(library(htmltools))

#Set working directory 
workdir<-"C:\\Wrksps\\dev\\DataScienceCE\\datasciencecoursera\\PracticalMachineLearning\\Project"
setwd(workdir)

trnsetFileName<-"pml-training.csv"
validtstsetFileName<-"pml-testing.csv"

dataUrlBase<-"https://d396qusza40orc.cloudfront.net/predmachlearn"

trnDataUrl<-paste(dataUrlBase,"/",trnsetFileName, sep="")

validtstDataUrl<-paste(dataUrlBase,"/",validtstsetFileName, sep="")


# Download file to local fie system

download.file(trnDataUrl,trnsetFileName)

download.file(validtstDataUrl,validtstsetFileName)

#Read the train, test/validation file contents to data frame

trnDataIn<-read.csv(trnsetFileName)

validtstDataIn<-read.csv(validtstsetFileName)

cat("The total number of rows/observations and columns/features of training data:",dim(trnDataIn)[1],dim(trnDataIn)[2],"\n")

cat("The total number of rows/observations and columns/features of test/validation data:",dim(validtstDataIn)[1],dim(validtstDataIn)[2],"\n")


vldtstFeatureNames<-names(validtstDataIn)

trnFeatureNames<-names(trnDataIn)

cat("Features:",trnFeatureNames)

head(trnDataIn)

trnFeaturesOnly<-setdiff(trnFeatureNames,vldtstFeatureNames)

cat("The feature in training but not in test/validation data set:",trnFeaturesOnly,"\n")

vldFeatursOnly<-setdiff(vldtstFeatureNames, trnFeatureNames)

cat("The feature in test/validation but not in training data set:",vldFeatursOnly,"\n")


#roll, pitch, yaw, total, kurtosis, skewness, max,min, amplitude, variance of total acceleration , average , standard 
#deviation , gyros_x,gyros_y,gyros_z, acceleration_X,acceleration_Y,acceleration_Z, magnetitude_x,magnetitude_y,magnetitude_z for all the accelerometer postions 
#belt, forearm, arm, and dumbell

#near zero variance

nZerov <- nearZeroVar(trnDataIn)

trnData1<-trnDataIn[,-nZerov]

cat("After removing near zero variance features, the total number of rows/observations and columns/features of training data:",dim(trnData1)[1],dim(trnData1)[2],"\n")


#NA values
naPercent    <- sapply(trnData1, function(x) mean(is.na(x))) > 0.95
trnDat2<-trnData1[,naPercent==FALSE]

cat("After removing features with more NA values, the total number of rows/observations and columns/features of training data:",dim(trnDat2)[1],dim(trnDat2)[2],"\n")

# Remove first 7 features specific to individuals and time stamps of excercise
trnDat<-trnDat2[,-c(1:7)]

cat("After removing features specific to individuals, the total number of rows/observations and columns/features of training data:",dim(trnDat)[1],dim(trnDat)[2],"\n")


outFeatureName<-c('classe')

trnFeatureName<-names(trnDat)

predictors<-setdiff(trnFeatureName, outFeatureName)

#rfe .. recursive feature elimination using rfe.. the following code is commented out becasue report generation taking long time. But while generating final report uncomment this #code.

#rfCtrl <- rfeControl(functions = rfFuncs,method = "repeatedcv",repeats = 1,verbose=TRUE,allowParallel = FALSE)

#trnDatProf <- rfe(trnDat[,predictors], trnDat[,outFeatureName], rfeControl = rfCtrl)

#cat(trnDatProf)

#seperate training data to train and test data sets based on classe factor variable

inTrain<-createDataPartition(y=trnDat$classe, p=0.75,list=FALSE)

training<-trnDat[inTrain,]

testing<-trnDat[-inTrain,]

trnames<-names(trnDat)

totl<-grepl("total_accel",trnames)

fnTotal<-trnames[which(totl==TRUE)]

featurePlot(x=training[, fnTotal], y=training$classe,plot="pairs",auto.key = list(columns = 3))

#featurePlot(x=training[, fnTotal], y=training$classe,plot="ellipse",auto.key = list(columns = 3))

featurePlot(x=training[, fnTotal], y=training$classe,plot="density",
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5, 
            pch = "|", 
            layout = c(4, 1), 
            auto.key = list(columns = 3))

rfeTop5<-c("yaw_belt", "magnet_dumbbell_z", "pitch_belt", "magnet_dumbbell_y", "pitch_forearm")

#rfeTop5<-trnDatProf$variables$var[1:5]

featurePlot(x=training[, rfeTop5], y=training$classe,plot="pairs",auto.key = list(columns = 5))

featurePlot(x=training[, rfeTop5], y=training$classe,plot="density",
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5, 
            pch = "|", 
            layout = c(5, 1), 
            auto.key = list(columns = 5))

# Classification tree

trControl <- trainControl(method="cv", number=5,verboseIter=FALSE,allowParallel = FALSE)

modelCLT <- train(classe~., data=training, method="rpart", trControl=trControl)

trainpred <- predict(modelCLT,newdata=testing)

#qplot(trainpred, classe, data=testing, xlab="predicted", ylab="Actual")

confMatCT <- confusionMatrix(testing$classe,trainpred)

confMatCT

fancyRpartPlot(modelCLT$finalModel)


#Random forest

controlRF <- trainControl(method="cv", number=3, verboseIter=TRUE,allowParallel = FALSE)

modelRF <- train(classe~., data=training, method="rf", trControl=controlRF, verbose=FALSE)

trainpredRF <- predict(modelRF,newdata=testing)

confMatRF <- confusionMatrix(testing$classe,trainpredRF)

confMatRF

#Boosting with RBM
modelGBM <- train(classe~., data=training, method="gbm", trControl=controlRF, verbose=FALSE)

trainpredGBM <- predict(modelGBM,newdata=testing)

confMatGBM <- confusionMatrix(testing$classe,trainpredGBM)

confMatGBM


#Final predtiction with validation data set

FinalTestPred <- predict(modelRF,newdata=validtstDataIn)

FinalTestPred

