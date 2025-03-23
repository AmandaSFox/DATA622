# Amanda Fox
# DATA 622 Assignment 2
# March 23, 2025

#---------------------------------
# Libraries
#---------------------------------

library(tidyverse)
library(caret)
library(ggplot2)
library(rpart)
library(rattle)
library(ROSE)
library(pROC)
library(doParallel)
library(fastAdaboost)
library(xgboost)
library(RColorBrewer)

#library(randomForest)
#library(themis)
#library(tidymodels)

#---------------------------------
# Preparations
#---------------------------------


# Load data
df <- read_csv("df_preprocessed.csv")
#df <- read_csv("https://raw.githubusercontent.com/AmandaSFox/DATA622/refs/heads/main/Assignment_2/df_preprocessed.csv")

# find highly correlated features (newly added features)
df_num <- df %>% 
  select(where(is.numeric))
cor_matrix <- df_num %>% 
  cor(use = "pairwise.complete.obs")
cor_df <- as.data.frame(as.table(cor_matrix)) %>%
  filter(Var1 != Var2) %>%                    
  filter(abs(Freq) > 0.1) %>%
  arrange(desc(abs(Freq)))

cor_df

# Remove two of three correlated features
df <- df %>% 
  select(-emp.var.rate, -nr.employed)

#Remove post hoc call duration feature
df <- df %>% 
  select(-Duration)

# Change character cols to factors
character_columns <- sapply(df, is.character)
df[character_columns] <- lapply(df[character_columns], as.factor)

glimpse(df)

summary(df)

# Initialize matrix to store metrics from each experiment
matrix_metrics <- matrix(NA, nrow = 0, ncol = 6) 
colnames(matrix_metrics) <- c("Model",
                              "Accuracy", 
                              "Precision", 
                              "Recall", 
                              "F1", 
                              "AUC")

# Turn on parallel processing (chatgpt)

num_cores <- detectCores() - 1 # Use one less core
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

#---------------------------------
# 1. Decision Tree: Unbalanced Y 
#---------------------------------

# Partition 70/30

set.seed(1)
arr_sample_unbal <- createDataPartition(y = df$Y, 
                                  p = .7, 
                                  list = FALSE)
df_unbal_train <- df[arr_sample_unbal, ]
df_unbal_test <- df[-arr_sample_unbal, ]

# Train first model
set.seed(123)
model_dt_unbal <- rpart(Y ~ ., 
                    method = "class",
                    data = df_unbal_train)

fancyRpartPlot(model_dt_unbal)

# remove classes not found in train & refactor
#df_unbal_test <- df_unbal_test[!(df_unbal_test$Occupation_Education %in% c("admin._illiterate", "housemaid_illiterate")), ]
#df_unbal_test$Occupation_Education <- factor(df_unbal_test$Occupation_Education)

# Test first model
predictions_unbal <- predict(model_dt_unbal, 
                              newdata = df_unbal_test, 
                              type = "class")

cm_unbal <- confusionMatrix(predictions_unbal, 
                            df_unbal_test$Y)

probabilities_unbal <- predict(model_dt_unbal, 
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

matrix_metrics

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

# Partition balanced data <- use for all remaining models
set.seed(2)
arr_sample_bal <- createDataPartition(y = df_bal$Y, 
                                      p = .7, 
                                      list = FALSE)
df_bal_train <- df_bal[arr_sample_bal, ]
df_bal_test <- df_bal[-arr_sample_bal, ]

# Train model
set.seed(234)
dt_bal <- rpart(Y ~ ., 
                  method = "class",
                  data = df_bal_train)

fancyRpartPlot(dt_bal)

# Test model
predictions_dt_bal <- predict(dt_bal, 
                             newdata = df_bal_test, 
                             type = "class")

cm_dt_bal <- confusionMatrix(predictions_dt_bal, 
                            df_bal_test$Y)

probabilities_dt_bal <- predict(dt_bal, 
                               newdata = df_bal_test, 
                               type = "prob")[, 2]

auc_dt_bal <- auc(roc(df_bal_test$Y, 
                     probabilities_dt_bal))

# Add metrics to the matrix
matrix_metrics <- rbind(matrix_metrics, 
                        c("2 Tree Balanced Y",
                          cm_dt_bal$overall["Accuracy"],
                          cm_dt_bal$byClass["Precision"],
                          cm_dt_bal$byClass["Recall"],
                          cm_dt_bal$byClass["F1"],
                          auc_dt_bal))

matrix_metrics

#---------------------------------
# 3. Random Forest: Default settings
#---------------------------------

# Using same train/test dataset as above (balanced Y)

# Train model (random forest package default: Random Forest model trained using the randomForest package with default settings. 
#This model was trained on the entire training dataset without any resampling, 
#cross-validation, or bootstrapping, resulting in a single Random Forest model)

set.seed(345)
rf_default <- randomForest::randomForest(Y~.,
                                         data = df_bal_train)

rf_default_predictions <- predict(rf_default, 
                                  df_bal_test)

# Test model
predictions_rf_bal <- predict(rf_default, 
                             newdata = df_bal_test, 
                             type = "class")

cm_rf_bal <- confusionMatrix(predictions_rf_bal, 
                            df_bal_test$Y)

probabilities_rf_bal <- predict(rf_default, 
                               newdata = df_bal_test, 
                               type = "prob")[, 2]

auc_rf_bal <- auc(roc(df_bal_test$Y, 
                     probabilities_rf_bal))

# Add metrics to the matrix
matrix_metrics <- rbind(matrix_metrics, 
                        c("3 Random Forest Default",
                          cm_rf_bal$overall["Accuracy"],
                          cm_rf_bal$byClass["Precision"],
                          cm_rf_bal$byClass["Recall"],
                          cm_rf_bal$byClass["F1"],
                          auc_rf_bal))

matrix_metrics


#---------------------------------
# 4. Random Forest: Hyperparameter tuned (grid)
#---------------------------------

# Define tuning grid for mtry values, five-fold cross-validation
rf_tuneGrid <- expand.grid(mtry=c(2,4,6,8,10,12)) #was 2,4,6,8
rf_control <- trainControl(method = "cv", number = 5)


# Train model (caret package with hyperparameter tuning)
set.seed(456)
rf_tuned <- caret::train(Y ~ ., 
                  data = df_bal_train, 
                  method = "rf", 
                  trControl = rf_control, 
                  tuneGrid = rf_tuneGrid)

# View caret's tuning selection results 
rf_tuned$results
rf_tuned$bestTune
rf_tuned$finalModel

# Test model
predictions_rf_tuned <- predict(rf_tuned, 
                              newdata = df_bal_test, 
                              type = "raw") #caret requires raw or prob

cm_rf_tuned <- confusionMatrix(predictions_rf_tuned, 
                             df_bal_test$Y)

probabilities_rf_tuned <- predict(rf_tuned, 
                                newdata = df_bal_test, 
                                type = "prob")[, 2]

auc_rf_tuned <- auc(roc(df_bal_test$Y, 
                      probabilities_rf_tuned))

# Add metrics to the matrix
matrix_metrics <- rbind(matrix_metrics, 
                        c("4 Random Forest Tuned",
                          cm_rf_tuned$overall["Accuracy"],
                          cm_rf_tuned$byClass["Precision"],
                          cm_rf_tuned$byClass["Recall"],
                          cm_rf_tuned$byClass["F1"],
                          auc_rf_tuned))

matrix_metrics

#---------------------------------
# 5. Adaboost: ada model errored out:  There were missing values in resampled performance measures.
#---------------------------------

#---------------------------------
# 5b. Adaboost: Hyperparameter tuned (grid)
#---------------------------------

# Additional data prep for adaboost (chatgpt)
ada_train <- droplevels(df_bal_train)
str(ada_train)

# Define tuning grid
# only one parameter tunable

ada_tuneGrid <- expand.grid(nIter = c(25,50,75),
                            method = "Adaboost.M1")

ada_control <- trainControl(method = "cv", 
                            number = 5,
                            classProbs = TRUE,
                            summaryFunction = twoClassSummary)


# Train model (fastAdaboost package with hyperparameter tuning)
set.seed(567)
ada_tuned <- caret::train(Y ~ ., 
                         data = ada_train, 
                         method = "adaboost", #fastAdaboost package 
                         trControl = ada_control, 
                         tuneGrid = ada_tuneGrid,
                         metric = "ROC",
                         preProcess = c("center","scale"))

# View caret's tuning selection results 
ada_tuned$bestTune
ada_tuned$results
ada_tuned$finalModel

# Test model
predictions_ada_tuned <- predict(ada_tuned, 
                                newdata = df_bal_test, 
                                type = "raw")

cm_ada_tuned <- confusionMatrix(predictions_ada_tuned, 
                               df_bal_test$Y)

probabilities_ada_tuned <- predict(ada_tuned, 
                                  newdata = df_bal_test, 
                                  type = "prob")[, 2]

auc_ada_tuned <- auc(roc(df_bal_test$Y, 
                        probabilities_ada_tuned))

# Add metrics to the matrix
matrix_metrics <- rbind(matrix_metrics, 
                        c("5 Adaboost Tuned",
                          cm_ada_tuned$overall["Accuracy"],
                          cm_ada_tuned$byClass["Precision"],
                          cm_ada_tuned$byClass["Recall"],
                          cm_ada_tuned$byClass["F1"],
                          auc_ada_tuned))

matrix_metrics

#---------------------------------
# 6. xgBoost: Hyperparameter tuned (grid)
#---------------------------------

xg_tuneGrid <- expand.grid(nrounds = 100, 
                           max_depth = c(3,6),
                           eta = c(0.1, 0.3),
                           gamma = 0,
                           colsample_bytree = 1,
                           min_child_weight = 1,
                           subsample = 1)

xg_control <- trainControl(method = "cv", 
                            number = 5,
                            classProbs = TRUE,
                            summaryFunction = twoClassSummary)

# Train model
set.seed(678)
xg_tuned <- caret::train(Y ~ ., 
                        data = ada_train, 
                        method = "xgbTree",  
                        trControl = xg_control, 
                        tuneGrid = xg_tuneGrid,
                        metric = "ROC",
                        preProcess = c("center","scale"))

# View caret's tuning selection results 
xg_tuned$bestTune
xg_tuned$results
xg_tuned$finalModel

# Test model
predictions_xg_tuned <- predict(xg_tuned, 
                                 newdata = df_bal_test, 
                                 type = "raw")

cm_xg_tuned <- confusionMatrix(predictions_xg_tuned, 
                                df_bal_test$Y)

probabilities_xg_tuned <- predict(xg_tuned, 
                                   newdata = df_bal_test, 
                                   type = "prob")[, 2]

auc_xg_tuned <- auc(roc(df_bal_test$Y, 
                         probabilities_xg_tuned))

# Add metrics to the matrix
matrix_metrics <- rbind(matrix_metrics, 
                        c("6 xgBoost Tuned",
                          cm_xg_tuned$overall["Accuracy"],
                          cm_xg_tuned$byClass["Precision"],
                          cm_xg_tuned$byClass["Recall"],
                          cm_xg_tuned$byClass["F1"],
                          auc_xg_tuned))

matrix_metrics

#---------------------------------
# 6b. xgBoost #2: Hyperparameter refined tuning (grid)
#---------------------------------

xg_tuneGrid_2 <- expand.grid(nrounds = c(100,200), 
                           max_depth = c(3,6),
                           eta = c(0.05, 0.1),
                           gamma = 0,
                           colsample_bytree = 1,
                           min_child_weight = 1,
                           subsample = 1)

xg_control <- trainControl(method = "cv", 
                           number = 5,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

#matrix_metrics <- matrix_metrics[matrix_metrics[, "Model"] != "7 xgBoost Tuned 2", ]

# Train model
set.seed(789)
xg_tuned2 <- caret::train(Y ~ ., 
                         data = ada_train, 
                         method = "xgbTree",  
                         trControl = xg_control, 
                         tuneGrid = xg_tuneGrid_2,
                         metric = "ROC",
                         preProcess = c("center","scale"))

# View caret's tuning selection results 
xg_tuned2$bestTune
xg_tuned2$results
xg_tuned2$finalModel

# Test model
predictions_xg_tuned2 <- predict(xg_tuned2, 
                                newdata = df_bal_test, 
                                type = "raw")

cm_xg_tuned2 <- confusionMatrix(predictions_xg_tuned2, 
                               df_bal_test$Y)

probabilities_xg_tuned2 <- predict(xg_tuned2, 
                                  newdata = df_bal_test, 
                                  type = "prob")[, 2]

auc_xg_tuned2 <- auc(roc(df_bal_test$Y, 
                        probabilities_xg_tuned2))

# Add metrics to the matrix
matrix_metrics <- rbind(matrix_metrics, 
                        c("7 xgBoost Tuned 2",
                          cm_xg_tuned2$overall["Accuracy"],
                          cm_xg_tuned2$byClass["Precision"],
                          cm_xg_tuned2$byClass["Recall"],
                          cm_xg_tuned2$byClass["F1"],
                          auc_xg_tuned2))

matrix_metrics


#---------------------------------
# parallel processing stop (chatgpt)
#---------------------------------

stopCluster(cl)
registerDoSEQ()


#---------------------------------
# Summary Charts for Essay
#---------------------------------

df_metrics <- as.data.frame(matrix_metrics)
df_metrics <- df_metrics %>%
  mutate(across(c(Accuracy, Precision, Recall, F1, AUC), 
                ~ round(as.numeric(.),4)))

print(df_metrics)

df_long <- df_metrics %>% 
  pivot_longer(cols = c(-Model),
               names_to = "Metric")

plot_compare <- df_long %>%  
  ggplot(aes(x = reorder(Model, desc(Model)),
             y = value,
             fill = Metric)) +
  geom_bar(stat = "identity", 
           show.legend = FALSE) +
  geom_text(aes(label = round(value, 3)), 
            hjust = -0.1, size = 2.5) +
  facet_wrap(~ Metric, 
               scales = "free_y",
               ncol = 1) +
  labs(title = "Model Performance by Metric",
       x = "",
       y = "")+
  coord_flip() +
  theme_minimal()
plot_compare  

