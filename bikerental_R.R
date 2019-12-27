rm(list = ls())
getwd()
setwd('C:/Users/Samruddhi/Desktop/Edwisor Project 2')
train= read.csv('day.csv')   #importing dataset

# Libraries Loading
x = c('tidyverse','lubridate','pedometrics','car','caret','randomForest','ggplot2','MLmetrics')
lapply(x, require, character.only = TRUE)

########################################## Exploratory Analysis #########################################

head(train)
str(train)
col_names=list(names(train))
for (i in col_names){
  print(summary(train[i]))
}

hist(train$weathersit, main= 'Histogram of wrking day')
hist(train$temp, main= 'Histogram of temp')
hist(train$atemp, main= 'Histogram of atemp')
hist(train$hum, main= 'Histogram of humidity')              # visualizing data distribution
hist(train$windspeed, main= 'Histogram of Wind Speed')
hist(train$casual, main= 'Histogram of casual')
hist(train$registered, main= 'Histogram of registered')

######################################### Feature Engineering ##########################################

train$dteday = format(as.Date(train$dteday, format='%d-%m-%Y'),format= '%d') # extracting days from date column
names(train)[2] = 'day'                      #rename 'dteday' variable by 'day'
train$day= as.numeric(train$day)             # converting into numeric datatype
head(train$day)
summary(train$day)
#write.csv(train,'newdata.csv',row.names = FALSE)
######################################### Missing Value Analysis ######################################

sum(is.na(train))       #No missing values present in our data set

######################################## Feature Selection ############################################

train_cat=train[c(2,3,4,5,6,7,8,9)]        # Subset of categorical variables
train_con= train[c(1,10,11,12,13,14,15,16)]  # Subset of continous variables

# Variance Inflation Factor Analysis
fit= lm(cnt~., data= train)
vif(fit)                        # calculating VIF values of each variable with threshold value of 5
train= train[-c(1)]             # removing 'instant' variable from the dataset
head(train)
train= train[-c(10)]            # removing 'atemp' variable
train= train[-c(12,13)]         # removing 'casual' and 'registered' variable
head(train)
######################################### Outlier Analysis ##############################################

boxplot(train$registered, main='Registered Users')
num_col= c('temp','hum','windspeed') #continous variables

for(i in num_col){
  print(i)
  val = train[,i][train[,i] %in% boxplot.stats(train[,i])$out]        # Removing Outliers using boxplot
  print(length(val))
  train = train[which(!train[,i] %in% val),]
}
length(train$day)
######################################## Model Selection ################################################
set.seed(123)
#Using K-fold cross validation method for model selection
# Linear Regression
LR= train(cnt~.,data=train , method='lm',trControl= trainControl(method= 'cv', number = 3) )
LR     # Linear Regression: RMSE: 876.639 Rsquared:0.794  MAE: 657.096

# KNN Algorithm
KNN= train(cnt~.,data=train , method='knn',trControl= trainControl(method= 'cv', number = 3))
KNN     # KNN Algorithm : RMSE: 1549.256  Rsquared:0.369  MAE: 1315.999

# Decision Tree
DT= train(cnt~.,data=train , method='rpart',trControl= trainControl(method= 'cv', number = 3))
DT      # Decision Tree: RMSE: 1104.027 Rsquared:0.675  MAE: 860.168

# Random Forest
RF= train(cnt~.,data=train , method='rf',trControl= trainControl(method= 'cv', number = 3))
RF      # Random Forest: RMSE: 649.3443 Rsquared:0.8887  MAE: 450.7936  for mtry:6

# Random Forest algorithm performs better than the above three algorithms for this dataset.

######################################## Model Development #############################################
# Hyper-Parameter Tuning

# finding best possible mtry value
RF= train(cnt~.,data=train , method='rf',trControl= trainControl(method= 'cv', number = 3))
RF
# optimum value of mtry= 6

# finding best possible ntree value
x= train[c(1,2,3,4,5,6,7,8,9,10,11)]  #independent variables
y= train[c(12)]                       # target variable
control = trainControl(method="cv", number=3, search="grid")
tunegrid = expand.grid(.mtry=6)
modellist = list()
for (ntree in c(500,600,700,800,900,1000, 1500, 2000, 2500,3000,3500,4000,4500,5000)) {
  set.seed(123)
  fit = train(cnt~., data=train, method="rf", tuneGrid=tunegrid, trControl=control, ntree=ntree)
  key = toString(ntree)
  modellist[[key]] = fit
}
results = resamples(modellist)
summary(results)
# For ntree= 1500, MAE= 438.341, RMSE= 622.9952, R-squared=  0.8653

# Model Building

set.seed(123)
train.index = createDataPartition(train$cnt, p = .80, list = FALSE)  # Data partition
train_1 = train[ train.index,]  # Dividing data into train and test sets
test_1  = train[-train.index,]

RF_model= randomForest(cnt~., train_1, mtry=6, ntree=1500, importance= TRUE)  
RF_predict= predict(RF_model,test_1[,-12])

# Model Performance Evaluation
mae = MAE(RF_predict,test_1[,12])             # MAE = 420.1074
mape = MAPE(RF_predict,test_1[,12])*100       # MAPE = 14.948 %
rmse=RMSE(RF_predict,test_1[,12])             # RMSE = 590.763
r2= R2(RF_predict,test_1[,12])                # R-squared = 0.912
