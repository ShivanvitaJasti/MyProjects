# importing the required libraries

library(tidyverse)
library(ggplot2)
library(RColorBrewer)
library(tidyr)
library(dplyr)
library(gridExtra)
library(stats)
library(grid)
library(usmap)
library(Metrics)
library(purrr)
library(leaps)
library(caTools)
library(psych)
library(maps)
library(Hmisc)
library(conf)
library(lattice)
library(caret)
library(glmnet)
library(ridge)
library(naivebayes)

# reading the dataset into a dataframe
Movie_data = read.csv("C:/Users/Public/Movie.csv")

# Viewing the first few records in the dataframe 
head(Movie_data)

# viewing the datatypes of attributes in the dataframe
str(Movie_data)

# checking for null values in the dataset
sum(is.na(Movie_data))

# removing the rows with the null values and checking the null values
Movie_data <- na.omit(Movie_data) 

sum(is.na(Movie_data))

# summarizing the dataset
summary(Movie_data)

# graphical representation of the relationships between the variables in the dataset
plot(Movie_data)

pairs.panels(Movie_data)

# Exploratory Data Analysis

# renaming certain columns in the dataframe
names(Movie_data)[names(Movie_data) == "Marketing.expense"] <- "Marketing"
names(Movie_data)[names(Movie_data) == "Production.expense"] <- "Production"
names(Movie_data)[names(Movie_data) == "Multiplex.coverage"] <- "Multiplex"
names(Movie_data)[names(Movie_data) == "Lead_.Actor_Rating"] <- "Actor_rating"
names(Movie_data)[names(Movie_data) == "Lead_Actress_rating"] <- "Actress_rating"
names(Movie_data)[names(Movie_data) == "Twitter_hastags"] <- "Twitter_Hashtags"

# checking the correlation of teh response variable with the remaining variables in the dataset
cor(Movie_data$Collection,Movie_data[c("Marketing","Production","Multiplex","Budget","Movie_length",
                                       "Actor_rating","Actress_rating","Director_rating","Producer_rating",
                                       "Critic_rating","Trailer_views","Time_taken",
                                       "Twitter_Hashtags","Avg_age_actors","Num_multiplex")])

# Proportions of categorical variables:
prop.table(table(Movie_data$X3D_available))

prop.table(table(Movie_data$Genre))

# Histograms for numeric variables:

hist.data.frame(Movie_data)

par(mfrow=c(2,2))

hist(Movie_data$Collection, main = "Distribution of Collections", xlab = "Collection" , border = "black", col = "lavender")

hist(Movie_data$Marketing, main = "Distribution of Marketing Expense", xlab = "Marketing Expense" , border = "black", col = "lavender")

hist(Movie_data$Production, main = "Distribution of Production Expense", xlab = "Production Expense" , border = "black", col = "lavender")

hist(Movie_data$Multiplex, main = "Distribution of Multiplex Coverage", xlab = "Multiplex Coverage" , border = "black", col = "lavender")

hist(Movie_data$Budget, main = "Distribution of Budget", xlab = "Buget" , border = "black", col = "lavender")

hist(Movie_data$Movie_length, main = "Distribution of Movie Length", xlab = "Movie Length" , border = "black", col = "lavender")

hist(Movie_data$Actor_rating, main = "Distribution of Actor Rating", xlab = "Actor Rating" , border = "black", col = "lavender")

hist(Movie_data$Actress_rating, main = "Distribution of Actress Rating", xlab = "Actress Rating" , border = "black", col = "lavender")

hist(Movie_data$Director_rating, main = "Distribution of Director Rating", xlab = "Director Rating" , border = "black", col = "lavender")

hist(Movie_data$Producer_rating, main = "Distribution of Producer Rating", xlab = "Producer Rating" , border = "black", col = "lavender")

hist(Movie_data$Critic_rating, main = "Distribution of Critic Rating", xlab = "Critic Rating" , border = "black", col = "lavender")

hist(Movie_data$Trailer_views, main = "Distribution of Trailer Views", xlab = "Trailer Views" , border = "black", col = "lavender")

hist(Movie_data$Time_taken, main = "Distribution of Time Taken", xlab = "Time Taken" , border = "black", col = "lavender")

hist(Movie_data$Twitter_Hashtags, main = "Distribution of Twitter Hashtags", xlab = "Twitter Hashtags" , border = "black", col = "lavender")

hist(Movie_data$Avg_age_actors, main = "Distribution of Actors' avarage age", xlab = "Actors' avarage age" , border = "black", col = "lavender")

hist(Movie_data$Num_multiplex, main = "Distribution of Number of Multiplexes", xlab = "Number of Multiplexes" , border = "black", col = "lavender")

par(mfrow=c(1,1))

# Boxplots for numeric variables:

par(mfrow=c(2,2))

boxplot(Movie_data$Collection, main = "Distribution of Collections", xlab = "Collection")
boxplot(Movie_data$Marketing, main = "Distribution of Marketing Expense", xlab = "Marketing Expense")
boxplot(Movie_data$Production, main = "Distribution of Production Expense", xlab = "Production Expense")
boxplot(Movie_data$Multiplex, main = "Distribution of Multiplex Coverage", xlab = "Multiplex Coverage" )
boxplot(Movie_data$Budget, main = "Distribution of Budget", xlab = "Buget")
boxplot(Movie_data$Movie_length, main = "Distribution of Movie Length", xlab = "Movie Length")
boxplot(Movie_data$Actor_rating, main = "Distribution of Actor Rating", xlab = "Actor Rating")
boxplot(Movie_data$Actress_rating, main = "Distribution of Actress Rating", xlab = "Actress Rating")
boxplot(Movie_data$Director_rating, main = "Distribution of Director Rating", xlab = "Director Rating")
boxplot(Movie_data$Producer_rating, main = "Distribution of Producer Rating", xlab = "Producer Rating")
boxplot(Movie_data$Critic_rating, main = "Distribution of Critic Rating", xlab = "Critic Rating")
boxplot(Movie_data$Trailer_views, main = "Distribution of Trailer Views", xlab = "Trailer Views")
boxplot(Movie_data$Time_taken, main = "Distribution of Time Taken", xlab = "Time Taken")
boxplot(Movie_data$Twitter_Hashtags, main = "Distribution of Twitter Hashtags", xlab = "Twitter Hashtags")
boxplot(Movie_data$Avg_age_actors, main = "Distribution of Actors' avarage age", xlab = "Actors' avarage age")
boxplot(Movie_data$Num_multiplex, main = "Distribution of Number of Multiplexes", xlab = "Number of Multiplexes")

par(mfrow=c(1,1))

# Bar plots for categorical variables:

ggplot(Movie_data, aes(x=reorder(Genre, Genre, function(x)-length(x)))) +
  geom_bar(fill='steelblue', color = "black", width = 0.2) +
  labs(x = "Genre") +
  geom_text(aes(label = ..count..), stat = "count", vjust = -0.7, colour = "black")

ggplot(Movie_data, aes(x=reorder(X3D_available, X3D_available, function(x)-length(x)))) +
  geom_bar(fill='cornsilk', color = "black", width = 0.2) +
  labs(x = "3D Availability") +
  geom_text(aes(label = ..count..), stat = "count", vjust = -0.7, colour = "black")

# Research Question - 1

# Is the success of a movie, which is measured by the Collections in this dataset, 
# affected by other features of a movie, namely the budget of the film, ratings given by critics, 
# number of views on the trailer, the expenses spent on Marketing and Production and so forth?

# Modeling - fitting a linear model to the dataset with all the numeric variables

linear_model <-lm(Collection ~ Marketing + Production + Multiplex + Budget + Movie_length + Actor_rating + 
                      Actress_rating + Director_rating + Producer_rating + Critic_rating + Trailer_views +
                      Time_taken + Twitter_Hashtags + Avg_age_actors + Num_multiplex,
                    data = Movie_data)

summary(linear_model)

par(mfrow=c(2,2))
plot(linear_model)
par(mfrow=c(1,1))

# The model with all the numerical variables accounts for 68.7 % of the variation in Collections

# Improving the model by taking the most significant variables in the variables
best_subset = regsubsets(Collection ~ Marketing + Production + Multiplex + Budget + Movie_length + Actor_rating + 
                           Actress_rating + Director_rating + Producer_rating + Critic_rating + Trailer_views +
                           Time_taken + Twitter_Hashtags + Avg_age_actors + Num_multiplex, data = Movie_data)

with(summary(best_subset),data.frame(rsq, adjr2, cp, rss, outmat))

# as we are looking for a model with the 
# least value of R2 , CP and RSS and the highest value of adjusted R2 value to fit the best model,
# selecting the model with 8 variables from the output table.

# with 8 variables:

model_8 <-lm(Collection ~ Marketing + Production + Multiplex + 
               Budget + Producer_rating + Critic_rating + Trailer_views + Time_taken,
             data = Movie_data)

summary(model_8)

par(mfrow=c(2,2))
plot(model_8)
par(mfrow=c(1,1))

coef(summary(model_8))

# Prediction using 0.95 Confidence Interval

movie_data_subset = Movie_data[14,]

movie_data_subset

predict(model_8, movie_data_subset, interval = "prediction", level = 0.95)

movie_data_subset = Movie_data[98,]

movie_data_subset

predict(model_8, movie_data_subset, interval = "prediction", level = 0.95)

movie_data_subset = Movie_data[201,]

movie_data_subset

predict(model_8, movie_data_subset, interval = "prediction", level = 0.95)

# Fitting a ridge regression model to the same data 

ridge_model <- linearRidge(Collection ~ Marketing + Production + Multiplex + 
                             Budget + Producer_rating + Critic_rating + Trailer_views + 
                             Time_taken,
                           data = Movie_data)  

predicted_values <- predict(ridge_model, Movie_data)

compared_values <- cbind (actual = Movie_data$Collection, predicted_values)  
compared_values

mean (apply(compared_values, 1, min)/apply(compared_values, 1, max))

# ridge regression - different method

df_rd = subset(Movie_data, select = -c(X3D_available))
df_rd = subset(df_rd, select = -c(Genre))

split <- sample.split(df_rd, SplitRatio = 0.8)
split

train <- subset(df_rd, split == "TRUE")
test <- subset(df_rd, split == "FALSE")

set.seed(1)
model_ridge <- train(
  Collection ~ .,
  data = train,
  method = 'ridge',
  preProcess = c("center", "scale")
)
model_ridge

test.values = subset(test, select=-c(Collection))
test.response = subset(test, select=Collection)[,1]

predictions = predict(model_ridge, newdata = test.values)

# RMSE
sqrt(mean((test.response - predictions)^2))

cor(test.response, predictions) ^ 2

plot(varImp(model_ridge))

# As can be seen, the collection can be approximately predicted from the selected predictor variables, 
# from the model and the collections fall in the range that is predicted by the model.

# Research Question - 2

# Can the expenses incurred on a movie be estimated from the collections made by the movie, 
# and other such variables?

# There are three types of expenses in this dataset: Budget, Production expense and Marketing expense

# Budget

linear_model_budget <-lm(Budget~ Marketing + Production + Multiplex + Collection + Movie_length + Actor_rating + 
                    Actress_rating + Director_rating + Producer_rating + Critic_rating + Trailer_views +
                    Time_taken + Twitter_Hashtags + Avg_age_actors + Num_multiplex,
                  data = Movie_data)

summary(linear_model_budget)

par(mfrow=c(2,2))
plot(linear_model)
par(mfrow=c(1,1))

best_subset_budget = regsubsets(Budget ~ Marketing + Production + Multiplex + Collection + Movie_length + Actor_rating + 
                           Actress_rating + Director_rating + Producer_rating + Critic_rating + Trailer_views +
                           Time_taken + Twitter_Hashtags + Avg_age_actors + Num_multiplex, data = Movie_data)

with(summary(best_subset_budget),data.frame(rsq, adjr2, cp, rss, outmat))

# 5 variables:
model_5 <-lm(Budget ~ Marketing + Production + Collection + 
               Movie_length + Trailer_views,
             data = Movie_data)

summary(model_5)

model_6 <-lm(Budget ~ Marketing + Production + Collection + 
               Movie_length + Trailer_views + Time_taken,
             data = Movie_data)

summary(model_6)

model_7 <-lm(Budget ~ Marketing + Production + Collection + 
               Movie_length + Trailer_views + Actor_rating + Director_rating,
             data = Movie_data)

summary(model_7)

par(mfrow=c(2,2))
plot(model_7)
par(mfrow=c(1,1))

# Prediction

movie_data_subset = Movie_data[14,]

movie_data_subset

predict(model_7, movie_data_subset, interval = "prediction", level = 0.95)

movie_data_subset = Movie_data[98,]

movie_data_subset

predict(model_7, movie_data_subset, interval = "prediction", level = 0.95)

movie_data_subset = Movie_data[201,]

movie_data_subset

predict(model_7, movie_data_subset, interval = "prediction", level = 0.95)

# Fitting a ridge regression model to the same data 

ridge_model_budget <- linearRidge(Budget ~ Marketing + Production + Collection + 
                             Movie_length + Trailer_views + Actor_rating + Director_rating,
                           data = Movie_data)  

predicted_values <- predict(ridge_model_budget, Movie_data)

compared_values <- cbind (actual = Movie_data$Budget, predicted_values)  
compared_values

mean (apply(compared_values, 1, min)/apply(compared_values, 1, max))

# ridge regression - different method

df_rd = subset(Movie_data, select = -c(X3D_available))
df_rd = subset(df_rd, select = -c(Genre))

split <- sample.split(df_rd, SplitRatio = 0.8)
split

train <- subset(df_rd, split == "TRUE")
test <- subset(df_rd, split == "FALSE")

set.seed(1)
model_ridge <- train(
  Budget ~ .,
  data = train,
  method = 'ridge',
  preProcess = c("center", "scale")
)
model_ridge

test.values = subset(test, select=-c(Budget))
test.response = subset(test, select=Budget)[,1]

predictions = predict(model_ridge, newdata = test.values)

# RMSE
sqrt(mean((test.response - predictions)^2))

cor(test.response, predictions) ^ 2

plot(varImp(model_ridge))


# lasso regression - different method

set.seed(1)
model_lasso <- train(
  Budget ~ .,
  data = train,
  method = 'lasso',
  preProcess = c("center", "scale")
)
model_lasso

test.values = subset(test, select=-c(Budget))
test.response = subset(test, select=Budget)[,1]

predictions = predict(model_lasso, newdata = test.values)

# RMSE
sqrt(mean((test.response - predictions)^2))

cor(test.response, predictions) ^ 2

# Production Expense

linear_model_production <-lm(Production ~ Marketing + Budget + Multiplex + Collection + Movie_length + Actor_rating + 
                           Actress_rating + Director_rating + Producer_rating + Critic_rating + Trailer_views +
                           Time_taken + Twitter_Hashtags + Avg_age_actors + Num_multiplex,
                         data = Movie_data)

summary(linear_model_production) 

best_subset_production = regsubsets(Production ~ Marketing + Budget + Multiplex + Collection + Movie_length + Actor_rating + 
                                  Actress_rating + Director_rating + Producer_rating + Critic_rating + Trailer_views +
                                  Time_taken + Twitter_Hashtags + Avg_age_actors + Num_multiplex,
                                data = Movie_data)

with(summary(best_subset_production),data.frame(rsq, adjr2, cp, rss, outmat))

model_7 <-lm(Production ~ Budget + Multiplex + Collection + 
               Director_rating + Trailer_views + Critic_rating + Time_taken,
             data = Movie_data)

summary(model_7)

par(mfrow=c(2,2))
plot(model_7)
par(mfrow=c(1,1))

# Prediction

movie_data_subset = Movie_data[14,]

movie_data_subset

predict(model_7, movie_data_subset, interval = "prediction", level = 0.95)

movie_data_subset = Movie_data[98,]

movie_data_subset

predict(model_7, movie_data_subset, interval = "prediction", level = 0.95)

movie_data_subset = Movie_data[201,]

movie_data_subset

predict(model_7, movie_data_subset, interval = "prediction", level = 0.95)

# Fitting a ridge regression model to the same data 

ridge_model_production <- linearRidge(Production ~ Budget + Multiplex + Collection + 
                                    Director_rating + Trailer_views + Critic_rating + Time_taken,
                                  data = Movie_data)

predicted_values <- predict(ridge_model_production, Movie_data)

compared_values <- cbind (actual = Movie_data$Production, predicted_values)  
compared_values

mean (apply(compared_values, 1, min)/apply(compared_values, 1, max))

# ridge regression - different method

df_rd = subset(Movie_data, select = -c(X3D_available))
df_rd = subset(df_rd, select = -c(Genre))

split <- sample.split(df_rd, SplitRatio = 0.8)
split

train <- subset(df_rd, split == "TRUE")
test <- subset(df_rd, split == "FALSE")

set.seed(1)
model_ridge <- train(
  Production ~ .,
  data = train,
  method = 'ridge',
  preProcess = c("center", "scale")
)
model_ridge

test.values = subset(test, select=-c(Production))
test.response = subset(test, select=Production)[,1]

predictions = predict(model_ridge, newdata = test.values)

# RMSE
sqrt(mean((test.response - predictions)^2))

cor(test.response, predictions) ^ 2

plot(varImp(model_ridge))

# lasso regression - different method

set.seed(1)
model_lasso <- train(
  Production ~ .,
  data = train,
  method = 'lasso',
  preProcess = c("center", "scale")
)
model_lasso

test.values = subset(test, select=-c(Production))
test.response = subset(test, select=Production)[,1]

predictions = predict(model_lasso, newdata = test.values)

# RMSE
sqrt(mean((test.response - predictions)^2))

cor(test.response, predictions) ^ 2

# Marketing Expense

linear_model_marketing <-lm(Marketing ~ Production+ Budget + Multiplex + Collection + Movie_length + Actor_rating + 
                               Actress_rating + Director_rating + Producer_rating + Critic_rating + Trailer_views +
                               Time_taken + Twitter_Hashtags + Avg_age_actors + Num_multiplex,
                             data = Movie_data)

summary(linear_model_marketing) 

best_subset_marketing = regsubsets(Marketing ~ Production + Budget + Multiplex + Collection + Movie_length + Actor_rating + 
                                      Actress_rating + Director_rating + Producer_rating + Critic_rating + Trailer_views +
                                      Time_taken + Twitter_Hashtags + Avg_age_actors + Num_multiplex,
                                    data = Movie_data)

with(summary(best_subset_marketing),data.frame(rsq, adjr2, cp, rss, outmat))

model_6 <-lm(Marketing ~ Budget + Multiplex + Collection + 
               Director_rating + Trailer_views + Time_taken,
             data = Movie_data)

summary(model_6)

par(mfrow=c(2,2))
plot(model_7)
par(mfrow=c(1,1))

model_7 <-lm(Marketing ~ Budget + Multiplex + Collection + 
               Movie_length + Director_rating + Trailer_views + Time_taken,
             data = Movie_data)

summary(model_7)

# Prediction

movie_data_subset = Movie_data[14,]

movie_data_subset

predict(model_6, movie_data_subset, interval = "prediction", level = 0.95)

movie_data_subset = Movie_data[98,]

movie_data_subset

predict(model_6, movie_data_subset, interval = "prediction", level = 0.95)

movie_data_subset = Movie_data[201,]

movie_data_subset

predict(model_6, movie_data_subset, interval = "prediction", level = 0.95)

# Fitting a ridge regression model to the same data 

ridge_model_marketing <- linearRidge(Marketing ~ Budget + Multiplex + Collection + 
                                        Movie_length + Director_rating + Trailer_views + Time_taken,
                                      data = Movie_data)

predicted_values <- predict(ridge_model_marketing, Movie_data)

compared_values <- cbind (actual = Movie_data$Marketing, predicted_values)  
compared_values

mean (apply(compared_values, 1, min)/apply(compared_values, 1, max))

# ridge regression - different method

df_rd = subset(Movie_data, select = -c(X3D_available))
df_rd = subset(df_rd, select = -c(Genre))

split <- sample.split(df_rd, SplitRatio = 0.8)
split

train <- subset(df_rd, split == "TRUE")
test <- subset(df_rd, split == "FALSE")

set.seed(1)
model_ridge <- train(
  Marketing ~ .,
  data = train,
  method = 'ridge',
  preProcess = c("center", "scale")
)
model_ridge

test.values = subset(test, select=-c(Marketing))
test.response = subset(test, select=Marketing)[,1]

predictions = predict(model_ridge, newdata = test.values)

# RMSE
sqrt(mean((test.response - predictions)^2))

cor(test.response, predictions) ^ 2

plot(varImp(model_ridge))

# lasso regression - different method

set.seed(1)
model_lasso <- train(
  Marketing ~ .,
  data = train,
  method = 'lasso',
  preProcess = c("center", "scale")
)
model_lasso

test.values = subset(test, select=-c(Marketing))
test.response = subset(test, select=Marketing)[,1]

predictions = predict(model_lasso, newdata = test.values)

# RMSE
sqrt(mean((test.response - predictions)^2))

cor(test.response, predictions) ^ 2


# Research Question - 3
# Does Budget depend on Genre, does a particular Genre incur more or less budget?

boxplot(Budget~Genre , 
        data = Movie_data ,
        main = "Budget per Genre" ,
        xlab ="Genre" ,
        ylab ="Budget per Genre")

# It can be seen that the budget across all the genres is more or less the same.

set.seed(1)
model_lasso <- train(
  Marketing.expense ~ .,
  data = train,
  method = 'lasso',
  preProcess = c("center", "scale")
)
model_lasso

test.values = subset(test, select=-c(Marketing.expense))
test.response = subset(test, select=Marketing.expense)[,1]

predictions = predict(model_lasso, newdata = test.values)

# RMSE
sqrt(mean((test.response - predictions)^2))

cor(test.response, predictions) ^ 2

# Research Question - 4 : Can a new picture be classified into a particular Genre based on the 
# movies already present, using the variables in the dataset as predictor variables?

# Naive Bayes

df_NB = subset(Movie_data, select = -c(X3D_available))

set.seed(100)

split <- sample.split(df_NB, SplitRatio = 0.8)
split

train <- subset(df_NB, split == "TRUE")
test <- subset(df_NB, split == "FALSE")

model_NB <- naive_bayes(Genre ~ ., data = train, usekernel = T) 

model_NB

plot(model_NB) 

predicted <- predict(model_NB, train, type = 'prob')
head(cbind(predicted, train))

# Confusion Matrix - train data

predict_train <- predict(model_NB, train)

table_train <- table(predict_train, train$Genre)

1 - sum(diag(table_train)) / sum(table_train)

# Confusion Matrix - test data

predict_test <- predict(model_NB, test)
table_test <- table(predict_test, test$Genre)

1 - sum(diag(table_test)) / sum(table_test)




