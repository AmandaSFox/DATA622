# Amanda Fox
# DATA 622 Assignment 1
# March 2, 2025

#---------------------------------
# Libraries
#---------------------------------

library(tidyverse)
library(ggplot2)
library(purrr)
library(knitr)
library(scales)
library(reshape2)
library(GGally)
library(RColorBrewer)

#---------------------------------
# Load Data
#---------------------------------
url <- "https://raw.githubusercontent.com/AmandaSFox/DATA622/refs/heads/main/Assignment_1/bank%2Bmarketing/bank-additional/bank-additional/bank-additional-full.csv"

#url <- "https://raw.githubusercontent.com/AmandaSFox/DATA622/refs/heads/main/Assignment_1/bank-full.csv"

df <- read_delim(url,
                 delim = ";",
                 quote = "\"")

#---------------------------------
# Overview
#---------------------------------

glimpse(df)

# missing data
colSums(is.na(df))

# duplicate records
nrow(df) - nrow(distinct(df))

# change col names based on documentation
df_final <- df %>% 
  set_names("Age","Occupation","Marital_Status","Education",
            "In_Default","Avg_Balance","Housing_Loan","Personal_Loan",
            "Contact_Type","Day","Month","Duration","Contacts_This_Campaign",
            "Days_Since_Last_Campaign","Contacts_Before_This_Campaign",
            "Previous_Outcome","Y")

#---------------------------------
# Categorical Variables
#---------------------------------

cat_cols <- df_final %>% 
  select(where(is.character)) %>%  
  names()

for (col in cat_cols) {
  df_final %>%
    count(.data[[col]]) %>%
    mutate(Pct = round((n / sum(n)) * 100, 1)) %>%
    arrange(desc(Pct)) %>% 
    print()
}

# Examples
plot_label <- df_final %>% 
  ggplot(aes(x = Y, fill = Y)) +
  geom_bar(fill="darkseagreen") +
  labs(title = "") +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 50000, by = 5000)) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.text.x = element_text(size = 14),
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        axis.text.y = element_text(size = 14))

print(plot_label)

plot_occupation <- df_final %>% 
  ggplot(aes(x = Occupation, fill = Y)) +
  geom_bar(fill="gold") +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 50000, by = 1000)) +
  labs(title = "",) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 45, hjust = 1, size = 14),
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        axis.text.y = element_text(size = 14))

print(plot_occupation)

#---------------------------------
# Numeric Variables
#---------------------------------

# create df of numeric variables except Day (not meaningful for correlations)
df_num <- df_final %>% 
  select(-Day) %>% 
  select(where(is.numeric))

#---------------------------------
# Age
summary(df_num$Age)

plot_age <- df_num %>% 
  ggplot(aes(Age))+
  geom_histogram(bins = 40,fill="deepskyblue4") +
  scale_x_continuous(breaks = seq(0, 100, by = 5)) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 50000, by = 500)) +
  labs(title = "Contact Age")+
  theme_minimal() +
  theme(legend.position = "none",
      title = element_text(size =16),
      axis.text.x = element_text(size = 14),
      axis.title.x = element_text(size = 16),
      axis.title.y = element_text(size = 16),
      axis.text.y = element_text(size = 14))

print(plot_age)

#---------------------------------
# Avg Balance
summary(df_num$Avg_Balance)

df_num %>% 
  ggplot(aes(x = Avg_Balance)) +
  geom_histogram(bins = 1000, fill = "gray80") +
  geom_vline(xintercept = 0, color="black", linetype = "dashed") +
  coord_cartesian(c(-1000,12000)) +
  scale_x_continuous(labels = scales::comma, 
                     breaks = seq(-1000, 12000, by = 2000)) +
  scale_y_continuous(labels = scales::comma, 
                     breaks = seq(0, 50000, by = 1000)) +
  labs(title = "Avg Annual Balance",
      subtitle = "Zoomed-In: Actual Max = 104K") +
  theme_minimal() +
  theme(legend.position = "none",
        title = element_text(size = 16),
        axis.text.x = element_text(size = 14),
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        axis.text.y = element_text(size = 14)) 

#---------------------------------
# Call Duration
summary(df_num$Duration)

ggplot(df_num, aes(x = Duration)) +
  geom_histogram(bins = 60, fill = "skyblue") +
  theme_minimal() +
  coord_cartesian(c(-100,2000)) +
  scale_x_continuous(breaks = seq(0, 2000, by = 200)) +
  labs(title = "Call Duration in Seconds: Zoomed")

#---------------------------------
# Contacts Before This Campaign
summary(df_num$Contacts_Before_This_Campaign)

df_final %>% 
  count(Contacts_Before_This_Campaign) %>% 
  #  mutate(Pct = round((n / sum(n)) * 100, 1))
  mutate(
    Contacts = ifelse(Contacts_Before_This_Campaign <= 5, 
                      as.character(Contacts_Before_This_Campaign), 
                      ">5"),
    Pct = round((n / sum(n)) * 100, 1)
  ) %>%
  group_by(Contacts) %>%
  summarise(n = sum(n), Pct = sum(Pct)) %>%
  arrange(as.numeric(Contacts)) %>% 
  print(n = Inf)  # Show all rows (helpful in case there's 0-9 and "Other")

#---------------------------------
# Contacts This Campaign
summary(df_num$Contacts_This_Campaign)

df_final %>% 
  count(Contacts_This_Campaign) %>% 
  #  mutate(Pct = round((n / sum(n)) * 100, 1))
  mutate(
    Contacts = ifelse(Contacts_This_Campaign <= 9, 
                      as.character(Contacts_This_Campaign), 
                      ">9"),
    Pct = round((n / sum(n)) * 100, 1)
  ) %>%
  group_by(Contacts) %>%
  summarise(n = sum(n), Pct = sum(Pct)) %>%
  arrange(as.numeric(Contacts)) %>% 
  print(n = Inf)  # Show all rows (helpful in case there's 0-9 and "Other")

#---------------------------------
# Days Since Last Campaign
summary(df_num$Days_Since_Last_Campaign)

df_final %>% 
  count(Days_Since_Last_Campaign) %>% 
  #  mutate(Pct = round((n / sum(n)) * 100, 1))
  mutate(
    Days = ifelse(Days_Since_Last_Campaign <= 5, 
                  as.character(Days_Since_Last_Campaign), 
                  ">5"),
    Pct = round((n / sum(n)) * 100, 1)
  ) %>%
  group_by(Days) %>%
  summarise(n = sum(n), Pct = sum(Pct)) %>%
  arrange(as.numeric(Days)) %>% 
  print()

# Remove -1 before charting
df_days_gt_0 <- df_num %>% 
  select(Days_Since_Last_Campaign) %>% 
  filter(Days_Since_Last_Campaign > 0)

ggplot(df_days_gt_0, aes(x = Days_Since_Last_Campaign)) +
  geom_histogram(bins = 60, fill = "skyblue") +
  theme_minimal() +
  coord_cartesian(c(-100,2000)) +
  scale_x_continuous(breaks = seq(0, 2000, by = 90)) +
  labs(title = "Days Since Last Campaign: >0 Days")

#---------------------------------
# Relationships Between Variables
#---------------------------------

# Numeric/Numeric

cor_matrix <- df_num %>% 
  cor(use = "pairwise.complete.obs")

# table of pairs with coefficients
cor_df <- as.data.frame(as.table(cor_matrix)) %>%
  filter(Var1 != Var2) %>%                    
  filter(abs(Freq) > 0.1) %>%
  arrange(desc(abs(Freq)))

cor_df

# plot
melt_cor <- melt(cor_matrix)

ggplot(melt_cor, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                       midpoint = 0, limit = c(-1,1), space = "Lab",
                       name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  coord_fixed()

#---------------------------------
# Categorical/Numeric Pairs 
# (ALL PAIRS - very long!)

#num_cols <- df_num %>% 
#  names()

#for (cat in cat_cols) {
#  for (num in num_cols) {
#    p <- ggplot(df_final, 
#                aes(x = .data[[cat]], 
#                    y = .data[[num]])) +
#      geom_boxplot(outlier.color = "red", outlier.shape = 1) +
#      theme_minimal() +
#      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
#      labs(title = paste(num, "by", cat))
    
#    print(p)
#  }
#}

# SAMPLE FOR ESSAY

# occupation by education
plot_ed_occ <- df_final %>% 
  ggplot(aes(x = Occupation, fill = Education)) +
    geom_bar(position = "fill") +
    labs(title = "Occupation by Education",
         x = "Occupation",
         y = "") +
    scale_y_continuous(labels = scales::percent_format(), breaks = pretty_breaks(6)) +
    theme_minimal() +
    theme(legend.position = "right",
          legend.text = element_text(size = 14),
          legend.title = element_text(size = 14),
          title = element_text(size = 16),
          axis.text.x = element_text(angle = 45, hjust = 1, size = 14),
          axis.title.x = element_text(size = 16),
          axis.title.y = element_text(size = 16),
          axis.text.y = element_text(size = 14))

print(plot_ed_occ)

# Y by occupation
plot_Y_occ <- df_final %>% 
  ggplot(aes(x = Occupation, fill = Y)) +
  geom_bar(position = "fill") +
  labs(title = "Occupation by Response (Y)",
       x = "Occupation",
       y = "") +
  scale_y_continuous(labels = scales::percent_format(), breaks = pretty_breaks(6)) +
  theme_minimal() +
  theme(legend.position = "right",
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 14),
        title = element_text(size = 16),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 14),
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        axis.text.y = element_text(size = 14))

print(plot_Y_occ)


# call duration by Y
plot_duration_y <- df_final %>%  
  ggplot(aes(x = Y, y = Duration, fill = Y)) +
  geom_boxplot(outlier.color = "red", outlier.shape = 1) +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "Call Duration by Response (Y)",
       x = "Response (Y)",
       y = "Call Duration (Seconds)") +
  coord_flip() +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(size = 16, face = "bold"),
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14))

print(plot_duration_y)


#---------------------------------
# Categorical/Categorical Pairs
# (ALL PAIRS - very long!)

#if (length(cat_cols) > 1) {
#  for (i in 1:(length(cat_cols) - 1)) {
#    for (j in (i + 1):length(cat_cols)) {
#      cat1 <- cat_cols[i]
#      cat2 <- cat_cols[j]
#      
#      p <- ggplot(df_final, 
#                  aes(x = .data[[cat1]], 
#                      fill = .data[[cat2]])) +
#        geom_bar(position = "fill") +
#        scale_y_continuous(labels = scales::label_percent()) +
#        labs(title = paste("Proportion of", cat2, "within", cat1),
#             y = "Proportion") +
#        theme_minimal() +
#        theme(axis.text.x = element_text(angle = 45, hjust = 1))
#      
#      print(p)
#    }
#  }
#}

#------------
# Pre-Processing
#------------

# Replace -1 with NA
df_preprocessed <- df_final %>%
  mutate(Days_Since_Last_Campaign = 
           ifelse(Days_Since_Last_Campaign == -1, 
                  NA, 
                  Days_Since_Last_Campaign))

# Results: Original and Revised
df_final %>% 
  count(Days_Since_Last_Campaign) %>% 
  mutate(
    Days = ifelse(Days_Since_Last_Campaign <= 5, 
                  as.character(Days_Since_Last_Campaign), 
                  ">5"),
    Pct = round((n / sum(n)) * 100, 1)
  ) %>%
  group_by(Days) %>%
  summarise(n = sum(n), Pct = sum(Pct)) %>%
  arrange(as.numeric(Days)) %>% 
  print()

df_preprocessed %>% 
  count(Days_Since_Last_Campaign) %>% 
  mutate(
    Days = ifelse(Days_Since_Last_Campaign <= 5, 
                  as.character(Days_Since_Last_Campaign), 
                  ">5"),
    Pct = round((n / sum(n)) * 100, 1)
  ) %>%
  group_by(Days) %>%
  summarise(n = sum(n), Pct = sum(Pct)) %>%
  arrange(as.numeric(Days)) %>% 
  print()

#---------------------------------
# occupation/education feature
df_preprocessed <- df_preprocessed %>%
  mutate(Occupation_Education = paste(Occupation, Education, sep = "_"))

df_preprocessed %>%
  count(Occupation_Education, sort = TRUE) %>%
  mutate(Pct = round((n / sum(n)) * 100, 1))

#---------------------------------
# loan profile feature: default, housing, personal loans
df_preprocessed <- df_preprocessed %>%
  mutate(Loan_Profile = case_when(
    Housing_Loan == "yes" & Personal_Loan == "yes" & In_Default == "yes" ~ "Both Loans - In Default",
    Housing_Loan == "yes" & Personal_Loan == "yes" & In_Default == "no" ~ "Both Loans - No Default",
    Housing_Loan == "yes" & Personal_Loan == "no" & In_Default == "yes" ~ "Housing Loan - In Default",
    Housing_Loan == "yes" & Personal_Loan == "no" & In_Default == "no" ~ "Housing Loan - No Default",
    Housing_Loan == "no" & Personal_Loan == "yes" & In_Default == "yes" ~ "Personal Loan - In Default",
    Housing_Loan == "no" & Personal_Loan == "yes" & In_Default == "no" ~ "Personal Loan - No Default",
    Housing_Loan == "no" & Personal_Loan == "no" & In_Default == "yes" ~ "No Loans - In Default",
    TRUE ~ "No Loans - No Default"
  ))

df_preprocessed %>%
  count(Loan_Profile, sort = TRUE) %>%
  mutate(Pct = round((n / sum(n)) * 100, 1))

