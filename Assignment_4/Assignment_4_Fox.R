# Assignment 4
# Predicting Churn in Telecommunications

#-----------------------------------
# Libraries
#-----------------------------------

library(tidyverse)
library(fastDummies)
library(randomForest)
library(pROC)
library(caret)
library(nnet)
library(scales)

#-----------------------------------
# Data Prep
#-----------------------------------

# Load data
df_raw <- read_csv("https://raw.githubusercontent.com/AmandaSFox/DATA622/refs/heads/main/Assignment_4/WA_Fn-UseC_-Telco-Customer-Churn.csv")

glimpse(df_raw)

# Missing values: 
colSums(is.na(df_raw)) #11 missing TotalCharges

missing_chgs <- df_raw %>% 
  filter(is.na(TotalCharges)) #these records have tenure = 0 and no charges
missing_chgs

df_raw <- df_raw %>%
  filter(!is.na(TotalCharges)) #removed records: tenure = 0, too new for churn model

# Duplicate rows
sum(duplicated(df_raw)) # no duplicate rows found

#--------------------------------
# Prepare Data for Modeling
#--------------------------------

# Standardize flags to numeric 0/1: 
# OK for trees and necessary for neural networks

# List of flag cols with char type and Yes/No values
yesno_cols <- c("Partner", "Dependents", "PhoneService", "MultipleLines",
                "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies",
                "PaperlessBilling", "Churn")  

# Original values in columns

yesno_counts_before <- df_raw %>%
  select(all_of(yesno_cols)) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value") %>%
  count(variable, value) %>%
  arrange(variable, desc(n)) %>% 
  pivot_wider(names_from = value, values_from = n)

yesno_counts_before

# Manual one-hot encoding:
# create new flag cols for "Phone Svc" and "Internet Svc" using values in existing cols

glimpse(df_raw)

df_model <- df_raw %>%
  # create binary flag for phone service
  mutate(Phone_Svc = if_else(PhoneService == "Yes", 1, 0)) %>%  
  # create binary flags for internet service and type
  mutate(Internet_Svc = if_else(InternetService == "No", 0, 1),
         DSL    = if_else(InternetService == "DSL", 1, 0),
         Fiber  = if_else(InternetService == "Fiber optic", 1, 0))

# verify new cols
df_model %>% 
  count(InternetService, Internet_Svc, DSL,Fiber)
df_model %>% 
  count(PhoneService, MultipleLines, Phone_Svc)

# remove old columns
df_model <- df_model %>% 
  select(-InternetService, -PhoneService)
  
# One-Hot encoding using package (gender, contract, and payment types)

# gender
df_model %>% 
  count(gender)

# contract
df_model %>%
  count(Contract)

# payment type
df_model %>%
  count(PaymentMethod)

df_model <- df_model %>%
  dummy_cols(
    select_columns = c("PaymentMethod", "Contract", "gender"),
    remove_selected_columns = TRUE
  )

# Clean up all binary flags to 0/1 (treat "No service" as 0)
df_model <- df_model %>% 
    mutate(MultipleLines = if_else(MultipleLines == "Yes", 1, 0),
           OnlineSecurity = if_else(OnlineSecurity == "Yes", 1, 0),
           OnlineBackup = if_else(OnlineBackup == "Yes", 1, 0),
           DeviceProtection = if_else(DeviceProtection == "Yes", 1, 0),
           TechSupport = if_else(TechSupport == "Yes", 1, 0),
           StreamingTV = if_else(StreamingTV == "Yes", 1, 0),
           StreamingMovies = if_else(StreamingMovies == "Yes", 1, 0),
           Partner = if_else(Partner == "Yes", 1, 0),
           Dependents = if_else(Dependents == "Yes", 1, 0),
           PaperlessBilling = if_else(PaperlessBilling == "Yes", 1, 0),
           Churn = if_else(Churn == "Yes", 1, 0))

glimpse(df_model)

# New values in flag cols

yesno_cols_after <- c("Partner", "Dependents", "MultipleLines",
                      "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                      "TechSupport", "StreamingTV", "StreamingMovies",
                      "PaperlessBilling", "Churn","Phone_Svc", "Internet_Svc",
                      "DSL","Fiber")

yesno_counts_after <- df_model %>%
  select(all_of(yesno_cols_after)) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value") %>%
  count(variable, value) %>%
  arrange(variable, desc(n)) %>% 
  pivot_wider(names_from = value, values_from = n) %>% 
  mutate(pct_yes = `1`/(`0`+`1`))

yesno_counts_after

# convert target variable to factor
df_model$Churn <- as.factor(df_model$Churn)

glimpse(df_model)

# summarize for paper
summary_stats <- df_model %>% 
  summarize(
    `Customers` = n(),
    `Churn %` = mean(Churn == 1),
    `Mean Tenure` = mean(tenure, na.rm = TRUE),
    `Mean Monthly Charges` = mean(MonthlyCharges, na.rm = TRUE),
    `% Month to Month` = mean(`Contract_Month-to-month` == 1), 
    `% Phone Services` = mean(Phone_Svc == 1),
    `% Internet Services` = mean(Internet_Svc == 1)) %>%
  pivot_longer(cols = everything(), names_to = "Metric", values_to = "Value")

summary_stats 

#-----------------------------------
# Modeling Preparation: Train/Test Data
#-----------------------------------

# initialize matrix to store metrics for all models
model_results <- matrix(nrow = 4, ncol = 7,
                        dimnames = list(
                          c("RF_default", "RF_tuned", "NN_simple", "NN_deep"),
                          c("Accuracy", "Precision", "Recall", "F1", "AUC", "Sensitivity","Specificity")
                        ))

# Train/test split 70/30
set.seed(123)

train_index <- sample(nrow(df_model), 0.7 * nrow(df_model))
df_train <- df_model[train_index, ]
df_test  <- df_model[-train_index, ]

# Drop customer ID
df_train <- df_train %>% select(-customerID)
df_test  <- df_test %>% select(-customerID)

# Make sure df_test has the same columns as df_train
missing_cols <- setdiff(names(df_train), names(df_test))
df_test[missing_cols] <- 0
df_test <- df_test[, names(df_train)]

missing_cols_rev <- setdiff(names(df_test), names(df_train))
df_train[missing_cols_rev] <- 0
df_train <- df_train[, names(df_test)]

# fix names with special characters for random forest (one hot encoded)
names(df_train) <- make.names(names(df_train))
names(df_test) <- make.names(names(df_test))

#-----------------------------------
# Model #1: random forest default parameters
#-----------------------------------

# Default Random Forest (note: exclude customerID from predictors)
rf_model <- randomForest(Churn ~ ., 
                         data = df_train, 
                         ntree = 500, 
                         importance = TRUE)

# Get predictions and probabilities
rf_preds <- predict(rf_model, newdata = df_test)
rf_probs <- predict(rf_model, newdata = df_test, type = "prob")[, 2]

# Confusion matrix
cm <- confusionMatrix(rf_preds, df_test$Churn, positive = "1")
cm

# AUC
rf_roc <- roc(df_test$Churn, rf_probs)
rf_auc <- auc(rf_roc)

# Store values in matrix
model_results["RF_default", "Accuracy"]  <- cm$overall["Accuracy"]
model_results["RF_default", "Precision"] <- cm$byClass["Precision"]
model_results["RF_default", "Recall"]    <- cm$byClass["Recall"]
model_results["RF_default", "F1"]        <- cm$byClass["F1"]
model_results["RF_default", "AUC"]       <- rf_auc
model_results["RF_default", "Sensitivity"] <- rf_tuned_cm$byClass["Sensitivity"]
model_results["RF_tuned", "Specificity"] <- rf_tuned_cm$byClass["Specificity"]


model_results


#-----------------------------------
# Model #2: random forest tuned parameters
#-----------------------------------

# Train with 5-fold cross-validation for mtry value
set.seed(42)

rf_tuned <- train(
  Churn ~ .,
  data = df_train,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5),
  tuneLength = 5
)

# Predictions
rf_tuned_preds <- predict(rf_tuned, newdata = df_test)
rf_tuned_probs <- predict(rf_tuned, newdata = df_test, type = "prob")[, 2]

# Evaluation
rf_tuned_cm <- confusionMatrix(rf_tuned_preds, df_test$Churn, positive = "1")
rf_tuned_roc <- roc(df_test$Churn, rf_tuned_probs)
rf_tuned_auc <- auc(rf_tuned_roc)

# Store in matrix
model_results["RF_tuned", "Accuracy"]    <- rf_tuned_cm$overall["Accuracy"]
model_results["RF_tuned", "Precision"]   <- rf_tuned_cm$byClass["Precision"]
model_results["RF_tuned", "Recall"]      <- rf_tuned_cm$byClass["Recall"]
model_results["RF_tuned", "F1"]          <- rf_tuned_cm$byClass["F1"]
model_results["RF_tuned", "AUC"]         <- rf_tuned_auc
model_results["RF_tuned", "Sensitivity"] <- rf_tuned_cm$byClass["Sensitivity"]
model_results["RF_tuned", "Specificity"] <- rf_tuned_cm$byClass["Specificity"]

# View
model_results

#-----------------------------------
# Model #3: simple NN
#-----------------------------------

# Convert Churn back to numeric 
y_train <- as.numeric(df_train$Churn)
y_test <- as.numeric(df_test$Churn)

# Drop target column and convert to matrix
x_train <- df_train %>% select(-Churn) %>% as.matrix()
x_test  <- df_test %>% select(-Churn) %>% as.matrix()

# Build model
nn_simple <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = ncol(x_train)) %>%
  layer_dense(units = 1, activation = "sigmoid")

# Compile
nn_simple %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = "accuracy"
)

# Fit model
history <- nn_simple %>% fit(
  x = x_train,
  y = y_train,
  epochs = 30,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 0
)

# Predict
nn_simple_probs <- predict(nn_simple, x_test)
nn_simple_preds <- ifelse(nn_simple_probs > 0.5, 1, 0)

# Evaluate
nn_simple_cm <- confusionMatrix(
  factor(nn_simple_preds, levels = c(0, 1)),
  factor(y_test, levels = c(0, 1)),
  positive = "1"
)
nn_simple_roc <- roc(y_test, nn_simple_probs)
nn_simple_auc <- auc(nn_simple_roc)

# Store metrics
model_results["NN_simple", "Accuracy"]    <- nn_simple_cm$overall["Accuracy"]
model_results["NN_simple", "Precision"]   <- nn_simple_cm$byClass["Precision"]
model_results["NN_simple", "Recall"]      <- nn_simple_cm$byClass["Recall"]
model_results["NN_simple", "F1"]          <- nn_simple_cm$byClass["F1"]
model_results["NN_simple", "AUC"]         <- nn_simple_auc
model_results["NN_simple", "Sensitivity"] <- nn_simple_cm$byClass["Sensitivity"]
model_results["NN_simple", "Specificity"] <- nn_simple_cm$byClass["Specificity"]
