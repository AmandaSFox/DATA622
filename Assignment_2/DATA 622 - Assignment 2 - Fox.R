# Amanda Fox
# DATA 622 Assignment 2
# March 23, 2025

# Experiment List: 
# Decision tree (3)
#   Feature engineering: 
#       unbalanced with all features, 
#       remove Duration (post hoc feature)
#       balance Y, 
# Random forest & adaboost (4)
#   Default settings (2)
#   Hyperparameter tuning - grid search (2) 
# Metamodel for ensemble method (1)


#---------------------------------
# Libraries
#---------------------------------

library(tidyverse)
library(ggplot2)
library(rpart)
library(rattle)
library(ROSE)
library(pROC)
library(randomForest)
library(themis)
library(tidymodels)
library(caret)

#library()
#library()
#library()

#---------------------------------
# Preparations
#---------------------------------

#df <- read_csv("https://raw.githubusercontent.com/AmandaSFox/DATA622/refs/heads/main/Assignment_2/df_preprocessed.csv")

# Load data
df_all <- read_csv("df_preprocessed.csv")
glimpse(df_all)

# ELABORATE ON THIS IN ESSAY
# since assignment #1, I changed to the dataset that included the external market features
# some of these were highly correlated and the below code identifies and removes those features
# (it also removes two features that I combined in assignment #1)

# find highly correlated features (newly added features)
df_num <- df_all %>% 
  select(where(is.numeric))
cor_matrix <- df_num %>% 
  cor(use = "pairwise.complete.obs")
cor_df <- as.data.frame(as.table(cor_matrix)) %>%
  filter(Var1 != Var2) %>%                    
  filter(abs(Freq) > 0.1) %>%
  arrange(desc(abs(Freq)))

cor_df

df_dur <- df_all %>% 
  select(-Housing_Loan, -Personal_Loan, -In_Default,
         -emp.var.rate, -nr.employed)
glimpse(df_dur)

# Change character cols to factors
character_columns <- sapply(df_dur, is.character)
df_dur[character_columns] <- lapply(df_dur[character_columns], as.factor)
glimpse(df_dur)

# Initialize matrix to store metrics from each experiment
matrix_metrics <- matrix(NA, nrow = 0, ncol = 6) 
colnames(matrix_metrics) <- c("Model",
                              "Accuracy", 
                              "Precision", 
                              "Recall", 
                              "F1", 
                              "AUC")

#---------------------------------
# 1. Decision Tree: Unbalanced Y, includes call duration 
#---------------------------------

# Partition 70/30

#set.seed(1)
arr_sample_dur <- createDataPartition(y = df_dur$Y, 
                                  p = .7, 
                                  list = FALSE)
df_unbal_train_dur <- df_dur[arr_sample_dur, ]
df_unbal_test_dur <- df_dur[-arr_sample_dur, ]

# Train first model
model_dt_unbal_dur <- rpart(Y ~ ., 
                    method = "class",
                    data = df_unbal_train_dur)

fancyRpartPlot(model_dt_unbal_dur)

# remove classes not found in train & refactor
#df_unbal_test <- df_unbal_test[!(df_unbal_test$Occupation_Education %in% c("admin._illiterate", "housemaid_illiterate")), ]
#df_unbal_test$Occupation_Education <- factor(df_unbal_test$Occupation_Education)


# Test first model
predictions_unbal_dur <- predict(model_dt_unbal_dur, 
                              newdata = df_unbal_test_dur, 
                              type = "class")

cm_unbal_dur <- confusionMatrix(predictions_unbal_dur, 
                            df_unbal_test_dur$Y)

probabilities_unbal_dur <- predict(model_dt_unbal_dur, 
                               newdata = df_unbal_test_dur, 
                               type = "prob")[, 2]

auc_unbal_dur <- auc(roc(df_unbal_test_dur$Y, 
                     probabilities_unbal_dur))

# Add metrics to the matrix
matrix_metrics <- rbind(matrix_metrics, 
                        c("1 Tree Unbalanced Y Duration",
                          cm_unbal_dur$overall["Accuracy"],
                          cm_unbal_dur$byClass["Precision"],
                          cm_unbal_dur$byClass["Recall"],
                          cm_unbal_dur$byClass["F1"],
                          auc_unbal_dur))

matrix_metrics

#---------------------------------
# 1b. Decision Tree: Unbalanced Y, excludes post hoc feature call duration
#---------------------------------

# EXPERIMENT: Remove duration feature
df_no_dur <- df_dur %>% 
  select(-Duration)
glimpse(df_no_dur)

# Partition 70/30

#set.seed(12)
arr_sample_no_dur <- createDataPartition(y = df_no_dur$Y, 
                                  p = .7, 
                                  list = FALSE)
df_unbal_train_no_dur <- df_no_dur[arr_sample_no_dur, ]
df_unbal_test_no_dur <- df_no_dur[-arr_sample_no_dur, ]

# Train first model. 
model_dt_unbal_no_dur <- rpart(
                        Y ~ ., 
                        data = df_unbal_train_no_dur,
                        method = "class")

fancyRpartPlot(model_dt_unbal_no_dur)

# remove classes not found in train & refactor
#df_unbal_test <- df_unbal_test[!(df_unbal_test$Occupation_Education %in% c("admin._illiterate", "housemaid_illiterate")), ]
#df_unbal_test$Occupation_Education <- factor(df_unbal_test$Occupation_Education)


# Test first model
predictions_unbal_no_dur <- predict(model_dt_unbal_no_dur, 
                             newdata = df_unbal_test_no_dur, 
                             type = "class")

cm_unbal_no_dur <- confusionMatrix(predictions_unbal_no_dur, 
                            df_unbal_test_no_dur$Y)

probabilities_unbal_no_dur <- predict(model_dt_unbal_no_dur, 
                               newdata = df_unbal_test_no_dur, 
                               type = "prob")[, 2]

auc_unbal_no_dur <- auc(roc(df_unbal_test_no_dur$Y, 
                     probabilities_unbal_no_dur))

# Add metrics to the matrix
matrix_metrics <- rbind(matrix_metrics, 
                        c("1b Tree Unbalanced Y Removed Duration",
                          cm_unbal_no_dur$overall["Accuracy"],
                          cm_unbal_no_dur$byClass["Precision"],
                          cm_unbal_no_dur$byClass["Recall"],
                          cm_unbal_no_dur$byClass["F1"],
                          auc_unbal_no_dur))

matrix_metrics

#---------------------------------
# 2. Decision Tree: Balanced (ROSE)
#---------------------------------

# show imbalance in Y
df_no_dur %>%
  count(Y) %>%
  mutate(percentage = n / sum(n) * 100)

# Balance using ROSE()
df_bal <- ROSE(Y ~ ., data = df_no_dur)$data

# Check the new balance
df_bal %>%
  count(Y) %>%
  mutate(percentage = n / sum(n) * 100)

# Make a new tree with balanced data

#set.seed(2)
arr_sample_bal <- createDataPartition(y = df_bal$Y, 
                                      p = .7, 
                                      list = FALSE)
df_bal_train <- df_bal[arr_sample_bal, ]
df_bal_test <- df_bal[-arr_sample_bal, ]

# Train model
dt_bal <- rpart(Y ~ ., 
                  method = "class",
                  data = df_bal_train)

fancyRpartPlot(dt_bal)

# Test model
predictions_bal <- predict(dt_bal, 
                             newdata = df_bal_test, 
                             type = "class")

cm_bal <- confusionMatrix(predictions_bal, 
                            df_bal_test$Y)

probabilities_bal <- predict(dt_bal, 
                               newdata = df_bal_test, 
                               type = "prob")[, 2]

auc_bal <- auc(roc(df_bal_test$Y, 
                     probabilities_bal))

# Add metrics to the matrix
matrix_metrics <- rbind(matrix_metrics, 
                        c("2 Tree Balanced Y",
                          cm_bal$overall["Accuracy"],
                          cm_bal$byClass["Precision"],
                          cm_bal$byClass["Recall"],
                          cm_bal$byClass["F1"],
                          auc_bal))

matrix_metrics

#---------------------------------
# 3. Random Forest: Default settings
#---------------------------------

# Using same train/test dataset as above (balanced Y)

# Train model
rf_default <- randomForest(Y ~ ., 
                           data = df_bal_train)


rf_default_predictions <- predict(rf_default, 
                                  df_bal_test)

# Test model
predictions_bal <- predict(dt_bal, 
                             newdata = df_bal_test, 
                             type = "class")

cm_bal <- confusionMatrix(predictions_bal, 
                            df_bal_test$Y)

probabilities_bal <- predict(dt_bal, 
                               newdata = df_bal_test, 
                               type = "prob")[, 2]

auc_bal <- auc(roc(df_bal_test$Y, 
                     probabilities_bal))

# Add metrics to the matrix
matrix_metrics <- rbind(matrix_metrics, 
                        c("2 Tree Balanced Y",
                          cm_bal$overall["Accuracy"],
                          cm_bal$byClass["Precision"],
                          cm_bal$byClass["Recall"],
                          cm_bal$byClass["F1"],
                          auc_bal))

matrix_metrics


#---------------------------------
# 4. Random Forest: Hyperparameter tuned (grid)
#---------------------------------

# Tuned Random Forest
# tuneGrid <- expand.grid(.mtry=c(2,3,4))
# rf_tuned <- train(Y ~ ., data = train_df, method = "rf", tuneGrid = tuneGrid)
# rf_tuned_predictions <- predict(rf_tuned, test_df)

# Evaluate your models using rf_default_predictions and rf_tuned_predictions


#---------------------------------
# 5. Adaboost: Balanced (SMOTE), hyperparameter tuned (grid)
#---------------------------------



#---------------------------------
# 6. Metamodel Ensemble: DT balanced, RF balanced/tuned, 
# Adaboost balanced/tuned
#---------------------------------
