library(tidyverse)
library(ggplot2)
library(purrr)
library(reshape2)
#library(corrplot)
library(GGally)


# dataset
df <- read_delim("bank-full.csv",
                 delim = ";",
                 quote = "\"")

# first look
glimpse(df)
summary(df)

#not used: str(df), dim(df), colnames(df), head(df)

# missing value check
colSums(is.na(df))

# prep data for charts 
df_final <- df %>% 
  set_names("Age","Occupation","Marital_Status","Education",
            "In_Default","Avg_Balance","Housing_Loan","Personal_Loan",
            "Contact_Type","Day","Month","Duration","Contacts_This_Campaign",
            "Days_Since_Last_Campaign","Contacts_Before_This_Campaign",
            "Previous_Outcome","Y")

# prep for charts: removed "Day" as it is not meaningful numerically
df_num <- df_final %>% 
  select(-Day) %>% 
  select(where(is.numeric))
df_long <- df_num %>% 
  pivot_longer(everything())

# histogram for distributions
plot_hist <- df_long %>% 
  ggplot(aes(value))+
    geom_histogram(bins = 30,fill="gray80") +
    facet_wrap(~name, scales = "free") +
    theme_minimal()
plot_hist

# contacts before this campaign
df_final %>% 
  count(Contacts_Before_This_Campaign)

# contacts this campaign
df_final %>% 
  count(Contacts_This_Campaign)

# Boxplot for outliers
plot_box <- df_long %>% 
  ggplot(aes(x = name, y = value)) +
  geom_boxplot(outlier.color = "red", outlier.shape = 1) +
  facet_wrap(~name, scales = "free") +
  theme_minimal()
plot_box

# Correlation matrix 
cor_matrix <- df_num %>% 
  cor(use = "pairwise.complete.obs")

melt_cor <- melt(cor_matrix)

ggplot(melt_cor, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                       midpoint = 0, limit = c(-1,1), space = "Lab",
                       name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  coord_fixed()

# table of pairs
cor_df <- as.data.frame(as.table(cor_matrix)) %>%
  filter(Var1 != Var2) %>%                    
  filter(abs(Freq) > 0.1) %>%
  arrange(desc(abs(Freq)))

cor_df

# scatterplot - not used, nothing is correlated

for (i in 1:nrow(cor_df)) {
  x_var <- cor_df$Var1[i]
  y_var <- cor_df$Var2[i]
  
  p <- ggplot(df, aes_string(x = x_var, y = y_var)) +
    geom_point(alpha = 0.6) +
    labs(title = paste("Scatterplot of", x_var, "vs", y_var, "\nCorrelation:", round(cor_df$Freq[i], 2))) +
    theme_minimal()
  
  print(p)
}


# note; the expanded dataset included financial indicators, which were
# correlated as follows: 
#         Var1           Var2      Freq
#1      euribor3m   emp.var.rate 0.9722447
#2   emp.var.rate      euribor3m 0.9722447
#3    nr.employed      euribor3m 0.9451544
#4      euribor3m    nr.employed 0.9451544
#5    nr.employed   emp.var.rate 0.9069701
#6   emp.var.rate    nr.employed 0.9069701
#7 cons.price.idx   emp.var.rate 0.7753342
#8   emp.var.rate cons.price.idx 0.7753342



# distributions for categorical columns, removed month because not meaningful
df_char <- select(df_final, -Month) %>%  
  select(where(is.character))
df_char_long <-df_char %>% 
  pivot_longer(everything()) %>%  
  count(name, value)
df_char_plot <- df_char_long %>% 
  ggplot(aes(x = reorder(value, n), y = n, fill = name)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~name, scales = "free") +
#  theme(axis.text.x = element_text(angle = 45, hjust = 1))
  coord_flip()
df_char_plot

# --------------
# scatterplots
# --------------

# numeric pairs - not used, just a mess, no correlations
pairs(df_num)

ggpair <- df_num %>%
  ggpairs(lower = list(continuous = "smooth"))

# num/char pairs
numeric_cols <- df_num %>% names()
categorical_cols <- df_char %>% %>% names()

for (cat in categorical_cols) {
  for (num in numeric_cols) {
    p <- ggplot(df, aes_string(x = cat, y = num)) +
      geom_boxplot(outlier.color = "red", outlier.shape = 1) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(title = paste("Boxplot of", num, "by", cat))
    
    print(p)  # Show each plot
  }
}


#char pairs

