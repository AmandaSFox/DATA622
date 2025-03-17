# Amanda Fox
# DATA 622 Assignment 2
# March 23, 2025

# Decision tree, random forest, adaboost
# Hyperparameter grid search (for each??) 
# metamodel for ensemble method


#---------------------------------
# Libraries
#---------------------------------

library(tidyverse)
library(ggplot2)
library(rpart)
library(rattle)
library(ROSE)
library(pROC)

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
df <- read_csv("df_preprocessed.csv")
glimpse(df)

# Change character cols to factors
character_columns <- sapply(df, is.character)
df[character_columns] <- lapply(df[character_columns], as.factor)
glimpse(df)

# Initialize matrix to store metrics from each experiment
matrix_metrics <- matrix(NA, nrow = 0, ncol = 6) 
colnames(matrix_metrics) <- c("Model",
                              "Accuracy", 
                              "Precision", 
                              "Recall", 
                              "F1", 
                              "AUC")

#---------------------------------
# 1. Decision Tree: Unbalanced Y, call duration with outliers
#---------------------------------

# Partition 70/30
#set.seed()
sample_set <- createDataPartition(y = df$Y, 
                                  p = .7, 
                                  list = FALSE)
df_unbal_train <- df[sample_set, ]
df_unbal_test <- df[-sample_set, ]

# Train first model
dt_unbal <- rpart(Y ~ ., 
                  method = "class",
                  data = df_unbal_train)

fancyRpartPlot(dt_unbal)

# remove classes not found in train & refactor
#df_unbal_test <- df_unbal_test[!(df_unbal_test$Occupation_Education %in% c("admin._illiterate", "housemaid_illiterate")), ]
#df_unbal_test$Occupation_Education <- factor(df_unbal_test$Occupation_Education)


# Test first model
predictions_unbal <- predict(dt_unbal, 
                              newdata = df_unbal_test, 
                              type = "class")

cm_unbal <- confusionMatrix(predictions_unbal, 
                            df_unbal_test$Y)

probabilities_unbal <- predict(dt_unbal, 
                               newdata = df_unbal_test, 
                               type = "prob")[, 2]

auc_unbal <- auc(roc(df_unbal_test$Y, 
                     probabilities_unbal))

# Add metrics to the matrix
matrix_metrics <- rbind(matrix_metrics, 
                        c("1 Tree Unbalanced Y",
                          cm_unbal$overall["Accuracy"],
                          cm_unbal$byClass["Precision"],
                          cm_unbal$byClass["Recall"],
                          cm_unbal$byClass["F1"],
                          auc_unbal))


#---------------------------------
# 2. Decision Tree: Balanced (ROSE)
#---------------------------------

# show imbalance in Y
df %>%
  count(Y) %>%
  mutate(percentage = n / sum(n) * 100)

# Balance using ROSE()
df_bal <- ROSE(Y ~ ., data = df)$data

# Check the new balance
df_bal %>%
  count(Y) %>%
  mutate(percentage = n / sum(n) * 100)

# Make a new tree
sample_set_bal <- createDataPartition(y = df_bal$Y, 
                                      p = .7, 
                                      list = FALSE)
df_bal_train <- df[sample_set_bal, ]
df_bal_test <- df[-sample_set_bal, ]

# Train model
dt_bal <- rpart(Y ~ ., 
                  method = "class",
                  data = df_bal_train)

summary(dt_bal)

fancyRpartPlot(dt_bal)

# Test model
predictions <- predict(dt_bal, newdata = df_bal_test, type = "class")

# Test first model
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

#---------------------------------
# 3. Random Forest: Default settings
#---------------------------------



#---------------------------------
# 4. Random Forest: Hyperparameter tuned (grid)
#---------------------------------



#---------------------------------
# 5. Adaboost: Balanced (SMOTE), hyperparameter tuned (grid)
#---------------------------------



#---------------------------------
# 6. Metamodel Ensemble: DT balanced, RF balanced/tuned, 
# Adaboost balanced/tuned
#---------------------------------
