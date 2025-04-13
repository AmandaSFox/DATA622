# Amanda Fox
# DATA 622 Assignment 3
# April 13, 2025

#---------------------------------
# Libraries
#---------------------------------

library(tidyverse)
library(caret)
library(ggplot2)
library(pROC)
library(doParallel)

#---------------------------------
# Load data from Assignment_2
#---------------------------------

# Balanced dataset and test/train split
df_bal <- read_csv("https://raw.githubusercontent.com/AmandaSFox/DATA622/refs/heads/main/Assignment_3/df_bal.csv")
df_bal_test <- read_csv("https://raw.githubusercontent.com/AmandaSFox/DATA622/refs/heads/main/Assignment_3/df_bal_test.csv")
df_bal_train <- read_csv("https://raw.githubusercontent.com/AmandaSFox/DATA622/refs/heads/main/Assignment_3/df_bal_train.csv")

# Matrix of metrics for each algorithm tested in Assignment 2 for comparison
df_metrics <- <- read_csv("https://raw.githubusercontent.com/AmandaSFox/DATA622/refs/heads/main/Assignment_3/df_metrics.csv")

#---------------------------------
# Prepare
#---------------------------------

# Initialize new matrix to store new metrics 
matrix_metrics <- matrix(NA, nrow = 0, ncol = 6) 
colnames(matrix_metrics) <- c("Model",
                              "Accuracy", 
                              "Precision", 
                              "Recall", 
                              "F1", 
                              "AUC")

# Turn on parallel processing for efficiency
num_cores <- detectCores() - 1 # Use one less core
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)










#---------------------------------
# Chart - Comparison of Metrics by Algorithm
#---------------------------------

# FIRST NEED TO APPEND NEW MATRIX OF METRICS TO DF_METRICS


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





#---------------------------------
# parallel processing stop (chatgpt)
#---------------------------------
stopCluster(cl)
registerDoSEQ()

