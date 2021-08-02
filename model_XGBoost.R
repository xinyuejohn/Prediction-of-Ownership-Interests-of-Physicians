library(tidyverse)
library(lubridate)
# data exploration
library(summarytools) # for user-friendly html summaries of data
library(ggmap) # for plotting data on a map
# for meta-ml
library(tidymodels)
library(dplyr)
library(ggpubr)
library(ranger)
library(xgboost)
library(caret)
physicians <- read_csv('physicians.csv')
companies <- read_csv('companies.csv')
payments <- read_csv('payments.csv')

#delete features we don't need
physicians <- physicians %>% rename(Physician_State = State) %>% rename(Physician_Country = Country)
physicians <- physicians %>% select(-First_Name,-Middle_Name,-Last_Name,-Physician_Country,-Name_Suffix, -Province,-Zipcode,-City,)
companies <- companies %>% rename(Company_State = State) %>% rename(Company_Country = Country)
companies <- companies %>% select(-Name)
payments <- payments %>% select(-Product_Name_1, -Product_Name_2, -Product_Name_3,
                                -Product_Category_2, -Product_Category_3,-Product_Code_1, -Product_Code_2, 
                                -Product_Code_3, -Contextual_Information,-City_of_Travel,-Date)

#get complete set
complete_set <- inner_join(payments, physicians, by = c("Physician_ID" = "id"))
complete_set <- inner_join(complete_set, companies, by = c("Company_ID" = "Company_ID"))

#get data for our training and test(train_and_test_data), 
#and also final test data we need to submit(predict_data)
train_and_test_data <- complete_set %>% filter(set=="train")
train_and_test_data <- train_and_test_data %>% select(-set)


predict_data <- complete_set %>% filter(set=="test")


#######################################################################################
#######################################################################################
#############################This is to get label for final test data###############
#######################################################################################


Ownership_Indicator_Predict <- predict_data %>% select(Ownership_Indicator)
Ownership_Indicator_Predict <- ifelse(Ownership_Indicator_Predict=="No",FALSE,TRUE)
ID_Predict <- predict_data %>% select(Physician_ID)
predict_data <- predict_data %>% select(-set,-Record_ID,-Ownership_Indicator)
predict_data_cache <- predict_data
#######################################################################################
#######################################################################################
#############################This is to get label for train and test data###############
#######################################################################################
#let the data be random
set.seed(1234)
train_and_test_data <- train_and_test_data[sample(1:nrow(train_and_test_data)), ]

#cache the data just in case
train_and_test_data_cache <- train_and_test_data

#get our label for training and testing data
Ownership_Indicator <- train_and_test_data %>% select(Ownership_Indicator)
Ownership_Indicator <- ifelse(Ownership_Indicator=="No",FALSE,TRUE)

#get our label with Physician_ID for training and testing data
Ownership_Indicator_ID <- train_and_test_data %>% select(Ownership_Indicator,Physician_ID)
Ownership_Indicator_ID$Ownership_Indicator <- ifelse(Ownership_Indicator_ID$Ownership_Indicator=="No",FALSE,TRUE)

#Delete the label from training and testing data
train_and_test_data <- train_and_test_data %>% select(-Ownership_Indicator,-Record_ID)
Ownership_Indicator_Cache <- Ownership_Indicator
numberOfSamples <- round(length(Ownership_Indicator_Cache) * .2)
#train_and_test_data[order(train_and_test_data$Physician_ID),]
#divide all training set into 5 subsets and train 5 models and finally get the average prediction
for (i in 0:4) {



   a = 1+i*numberOfSamples
   b = numberOfSamples*(i+1) 
   
train_and_test_data_iteration <- train_and_test_data[a:b,]
Ownership_Indicator <- Ownership_Indicator_Cache[a:b,]
train_and_test_labels_ID <- Ownership_Indicator_ID[a:b,]


   
predict_data <- predict_data_cache
train_and_test_data_iteration <- rbind(predict_data,train_and_test_data_iteration)
train_and_test_data_iteration <- train_and_test_data_iteration %>% select(-Physician_ID,-Company_ID)

#######################################################################################
#######################################################################################
#############################This is training data process#############################
#######################################################################################
options(na.action="na.pass")


form_of_payment <- model.matrix(~Form_of_Payment_or_Transfer_of_Value-1,train_and_test_data_iteration)

nature_of_payment <- model.matrix(~Nature_of_Payment_or_Transfer_of_Value-1,train_and_test_data_iteration)

related_product_indicator <- model.matrix(~Related_Product_Indicator-1,train_and_test_data_iteration)


product_type_1 <- model.matrix(~Product_Type_1-1,train_and_test_data_iteration)
product_type_2 <- model.matrix(~Product_Type_2-1,train_and_test_data_iteration)
product_type_3 <- model.matrix(~Product_Type_3-1,train_and_test_data_iteration)

physician_state <- model.matrix(~Physician_State-1,train_and_test_data_iteration)

license_state_1 <- model.matrix(~License_State_1-1,train_and_test_data_iteration)
license_state_2 <- model.matrix(~License_State_2-1,train_and_test_data_iteration)
license_state_3 <- model.matrix(~License_State_3-1,train_and_test_data_iteration)


company_state <- model.matrix(~Company_State-1,train_and_test_data_iteration)




train_and_test_data_iteration_numeric <- train_and_test_data_iteration %>% select_if(is.numeric)
rm(train_and_test_data_iteration)
train_and_test_data_iteration_numeric <- cbind(train_and_test_data_iteration_numeric,
                                     form_of_payment,
                                     nature_of_payment,
                                     related_product_indicator,
                                     product_type_1,
                                     product_type_2,
                                     product_type_3,
                                     physician_state,
                                     license_state_1,
                                     license_state_2,
                                     license_state_3,
                                     company_state)

rm(form_of_payment,
   nature_of_payment,
   related_product_indicator,
   product_type_1,
   product_type_2,
   product_type_3,
   physician_state,
   license_state_1,
   license_state_2,
   license_state_3,
   company_state
   )

train_and_test_data_iteration_matrix <- data.matrix(train_and_test_data_iteration_numeric)
rm(train_and_test_data_iteration_numeric)





#divide data into training and test data

predict_data <- train_and_test_data_iteration_matrix[1:length(Ownership_Indicator_Predict),]
predict_labels <- Ownership_Indicator_Predict
dpredict <- xgb.DMatrix(data = predict_data, label= predict_labels)
train_and_test_data_iteration_matrix <- train_and_test_data_iteration_matrix[-(1:length(Ownership_Indicator_Predict)),]
   



numberOfTrainingSamples <- round(length(Ownership_Indicator) * .7)

train_data <- train_and_test_data_iteration_matrix[1:numberOfTrainingSamples,]
train_labels <- Ownership_Indicator[1:numberOfTrainingSamples]

# testing data
test_data <- train_and_test_data_iteration_matrix[-(1:numberOfTrainingSamples),]
test_labels <- Ownership_Indicator[-(1:numberOfTrainingSamples)]
test_labels_ID <- train_and_test_labels_ID[-(1:numberOfTrainingSamples),]


rm(train_and_test_data_iteration_matrix)

dtrain <- xgb.DMatrix(data = train_data, label= train_labels)
dtest <- xgb.DMatrix(data = test_data, label= test_labels)

negative_cases <- sum(train_labels == FALSE)
postive_cases <- sum(train_labels == TRUE)



###################Training the model###################
#########################################################
#########################################################

# n_round <- 50 #round of iterations
# early_stopping_rounds <- 10
if(i==0){
   watchlist <- list(validation=dtest, train=dtrain)
   param <- list(
      objective = "binary:logistic",
      max_depth =  10,
      eta = 0.4
      
   )
   
   model1 <- xgb.train(data = dtrain, # the data   
                 params = param,
                 nround = 120, 
                 scale_pos_weight = negative_cases/postive_cases,
                 watchlist = watchlist
                 )
   test_1 <- predict(model1, dtest)
   test_result_1 <- bind_cols(test_labels_ID,test_1,.id = NULL)
   pred_1 <- predict(model1, dpredict)

}






if(i==1){
   watchlist <- list(validation=dtest, train=dtrain)
   param <- list(
      objective = "binary:logistic",
      max_depth =  10,
      eta = 0.4
   )
   
   model2 <- xgb.train(data = dtrain, # the data   
                       params = param,
                       nround = 120, 
                       scale_pos_weight = negative_cases/postive_cases,
                       watchlist = watchlist
   )
   test_2 <- predict(model2, dtest)
   test_result_2 <- bind_cols(test_labels_ID,test_2,.id = NULL)
   pred_2 <- predict(model2, dpredict)
}

if(i==2){
   watchlist <- list(validation=dtest, train=dtrain)
   param <- list(
      objective = "binary:logistic",
      max_depth = 10,
      eta = 0.4 
   )
   
   model3 <- xgb.train(data = dtrain, # the data   
                       params = param,
                       nround = 120, 
                       scale_pos_weight = negative_cases/postive_cases,
                       watchlist = watchlist
                       
   )
   test_3 <- predict(model3, dtest)
   test_result_3 <- bind_cols(test_labels_ID,test_3,.id = NULL)
   pred_3 <- predict(model3, dpredict)
}

if(i==3){
   watchlist <- list(validation=dtest, train=dtrain)
   param <- list(
      objective = "binary:logistic",
      max_depth =  10,
      eta = 0.4
   )
   
   model4 <- xgb.train(data = dtrain, # the data   
                       params = param,
                       nround = 120, 
                       scale_pos_weight = negative_cases/postive_cases,
                       watchlist = watchlist
                       
   )
   test_4 <- predict(model4, dtest)
   test_result_4 <- bind_cols(test_labels_ID,test_4,.id = NULL)
   pred_4 <- predict(model4, dpredict)
}

if(i==4){
   watchlist <- list(validation=dtest, train=dtrain)
   param <- list(
      objective = "binary:logistic",
      max_depth = 10,
      eta = 0.4
   )
   
   model5 <- xgb.train(data = dtrain, # the data   
                       params = param,
                       nround = 120, 
                       scale_pos_weight = negative_cases/postive_cases,
                       watchlist = watchlist
                       
   )
   test_5 <- predict(model5, dtest)
   test_result_5 <- bind_cols(test_labels_ID,test_5,.id = NULL)
   pred_5 <- predict(model5, dpredict)
}


rm(train_data,test_data,predict_data)

}







#########################################################
#########################################################
#             Predict over final test data
#########################################################
#########################################################


threshold <- 0.02
pred_1 <- pred_1_cache
pred_2<- pred_2_cache
pred_3 <- pred_3_cache
pred_4 <- pred_4_cache
pred_5 <- pred_5_cache



pred_1 <- ifelse(pred_1>threshold,TRUE,FALSE)
pred_2 <- ifelse(pred_2>threshold,TRUE,FALSE)
pred_3 <- ifelse(pred_3>threshold,TRUE,FALSE)
pred_4 <- ifelse(pred_4>threshold,TRUE,FALSE)
pred_5 <- ifelse(pred_5>threshold,TRUE,FALSE)



pred_result <- bind_cols(ID_Predict,pred_1,pred_2,pred_3,pred_4,pred_5,.id = NULL)
names(pred_result)[2] <- paste("Prediction_1")
names(pred_result)[3] <- paste("Prediction_2")
names(pred_result)[4] <- paste("Prediction_3")
names(pred_result)[5] <- paste("Prediction_4")
names(pred_result)[6] <- paste("Prediction_5")

pred_result <- pred_result %>% group_by(Physician_ID) %>% 
   mutate(Prediction_Whole_1=if(any(Prediction_1 == TRUE)) {1} else {0})
pred_result <- pred_result %>% select(-Prediction_1)

pred_result <- pred_result %>% group_by(Physician_ID) %>% 
   mutate(Prediction_Whole_2=if(any(Prediction_2 == TRUE)) {1} else {0})
pred_result <- pred_result %>% select(-Prediction_2)

pred_result <- pred_result %>% group_by(Physician_ID) %>% 
   mutate(Prediction_Whole_3=if(any(Prediction_3 == TRUE)) {1} else {0})
pred_result <- pred_result %>% select(-Prediction_3)

pred_result <- pred_result %>% group_by(Physician_ID) %>% 
   mutate(Prediction_Whole_4=if(any(Prediction_4 == TRUE)) {1} else {0})
pred_result <- pred_result %>% select(-Prediction_4)

pred_result <- pred_result %>% group_by(Physician_ID) %>% 
   mutate(Prediction_Whole_5=if(any(Prediction_5 == TRUE)) {1} else {0})
pred_result <- pred_result %>% select(-Prediction_5)


pred_result <- unique(pred_result,by="Physician_ID")

pred_result[order(pred_result$Physician_ID),]
##############################################
#needs select majority
pred_result <- pred_result %>% mutate(
   prediction_sum = Prediction_Whole_1 +Prediction_Whole_2 +Prediction_Whole_3 + Prediction_Whole_4 +Prediction_Whole_5
)



pred_result <- pred_result %>% mutate(
   prediction = ifelse(prediction_sum>=3,1,0)
)

pred_result <- pred_result %>% select(-Prediction_Whole_1,-Prediction_Whole_2,-Prediction_Whole_3,-Prediction_Whole_4,-Prediction_Whole_5)
pred_result <- pred_result %>% select(-prediction_sum)
##############################################

names(pred_result)[1] <- paste("id")
names(pred_result)[2] <- paste("prediction")
write_csv(pred_result, "SUBMISSION.csv")
