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
Student_data = read.csv("C:/Users/Public/Cleaned_Student_data.csv")

# Viewing the first few records in the dataframe 
head(Student_data)

# viewing the datatypes of attributes in the dataframe
str(Student_data)

# checking for null values in the dataset
sum(is.na(Student_data))

# summarizing the dataset
summary(Student_data)

# proportions of values in Categorical columns

prop.table(table(Student_data$Certifications))

prop.table(table(Student_data$hobbies))

prop.table(table(Student_data$Prefer_study_time))

prop.table(table(Student_data$Interest))

## MODELING

# all the variables

linear_model <-lm(college.mark ~ Certifications + X10th.Mark + X12th.Mark + hobbies + Daily_study_time + Prefer_study_time + Exp_salary + Interest + Career_basedon_degree + Stress.Level + Financial.Status,
                    data = Student_data)

summary(linear_model)

par(mfrow=c(2,2))
plot(linear_model)
par(mfrow=c(1,1))

# to determine best subset

best_subset = regsubsets(college.mark ~ Certifications + X10th.Mark + X12th.Mark + hobbies + Daily_study_time + Prefer_study_time + Exp_salary + Interest + Career_basedon_degree + Stress.Level + Financial.Status,
                         data = Student_data)

# to determine the best subset of predictor variables to fit the regression model
with(summary(best_subset),data.frame(rsq, adjr2, cp, rss, outmat))

# with best subset variables:

linear_model <-lm(college.mark ~ Certifications + X10th.Mark + X12th.Mark + Prefer_study_time + Interest + Career_basedon_degree + Stress.Level,
                  data = Student_data)

summary(linear_model)

par(mfrow=c(2,2))
plot(linear_model)
par(mfrow=c(1,1))

# 1) Does the hobby 'reading books' have a greater positive effect on College marks as opposed to other hobbies

ggplot(Student_data, aes(x = college.mark, y = hobbies)) +
  geom_point(aes(x = college.mark, colour = "college.mark" , group = 1),size = 3) +
  geom_line() + 
  ggtitle("Plot of College Marks and Hobbies") +
  xlab("College Marks") + ylab("Hobbies")

# 2) Is there a correlation between 10th, 12th and college marks of a student?

#-----------------
ggplot(data=Student_data, aes(x=X10th.Mark, y=college.mark, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(colour="blue") +
  ggtitle("Correlation between 10th and College marks of the students") +
  xlab("10th Marks") + ylab("College Marks")

plot(x = Student_data$X10th.Mark, y = Student_data$college.mark, col="blue", pch = 19,
     xlab = "10th Marks",
     ylab = "College Marks",
     main = "Correlation between 10th and College marks of the students"
) 
abline(lm(college.mark~X10th.Mark,data=Student_data),col='red')

#--------------------

ggplot(data=Student_data, aes(x=X12th.Mark, y=college.mark, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point() +
  ggtitle("Correlation between 12th and College marks of the students") +
  xlab("12th Marks") + ylab("College Marks")

plot(x = Student_data$X12th.Mark, y = Student_data$college.mark, col="dark green", pch = 19,
     xlab = "12th Marks",
     ylab = "College Marks",
     main = "Correlation between 12th and College marks of the students"
) 
abline(lm(college.mark~X12th.Mark,data=Student_data),col='red')

#---------------------

ggplot(data=Student_data, aes(x=X10th.Mark, y=X12th.Mark, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point() +
  ggtitle("Correlation between 10th and 12th marks of the students") +
  xlab("10th Marks") + ylab("12th Marks")

plot(x = Student_data$X10th.Mark, y = Student_data$X12th.Mark, col="purple", pch = 19,
     xlab = "10th Marks",
     ylab = "12th Marks",
     main = "Correlation between 10th and 12th marks of the students"
) 
abline(lm(X12th.Mark~X10th.Mark,data=Student_data),col='red')

#----------------

# 3) Does daily study time of a student have an impact on their college marks?

ggplot(data=Student_data, aes(x=Daily_study_time, y=college.mark, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(col="blue") +
  ggtitle("Relation between the daily study time and college marks of students") +
  xlab("Daily Study Time(minutes)") + ylab("College Marks")

boxplot( college.mark~Daily_study_time,
         data=Student_data,
         main="Relation between the Daily Study time and college marks of a student",
         xlab="Daily Study Time (minutes)",
         ylab="College Marks",
         col="lavender",
         border="black"
)

# 4) Does the preferred time of day to study have an impact on a student's college marks?

ggplot(data=Student_data, aes(x=Prefer_study_time, y=college.mark, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(col="dark green") +
  ggtitle("Relation between the Preferred study time and college marks of students") +
  xlab("Preferred Study Time") + ylab("College Marks")

boxplot( college.mark~Prefer_study_time,
         data=Student_data,
         main="Relation between the preferred time of study and college marks of a student",
         xlab="Preferred Study Time",
         ylab="College Marks",
         col="lavender",
         border="black"
)

# 5) Does a student liking their degree have an impact on the student choosing their career based on the degree?

mosaicplot(Career_basedon_degree~Interest,data=Student_data, col="skyblue",
           main = "Career based on degree v/s student's interest in the degree", 
           xlab = "Career based on Degree", y = "Interest")

boxplot( Career_basedon_degree~Interest,
         data=Student_data,
         main="Relation between the Probability of student's choosing a career based on their degree and \n their interest in the degree",
         xlab="Interest",
         ylab="Career based on the degree",
         col="lavender",
         border="black"
)

# 6) Does the completion of certification courses have an impact on the expected salaries of a student?

ggplot(Student_data, aes(x = Exp_salary, y = Certifications)) +
  geom_point(aes(x = Exp_salary, colour = "Exp_salary" , group = 1),size = 3) +
  geom_line() + 
  ggtitle("Plot of Completion of Certifications and the student's expected salary") +
  xlab("Expected Salary") + ylab("Certifications")

ggplot(Student_data, aes(x = Exp_salary, y = Certifications)) +
  geom_point(aes(x = Exp_salary, colour = "Exp_salary" , group = 1),size = 3) +
  geom_line() + 
  xlim(0, 150000) +
  ggtitle("Plot of Completion of Certifications and the student's expected salary") +
  xlab("Expected Salary") + ylab("Certifications")

# 7) Does financial status have an effect on the stress levels of a student?

mosaicplot(Stress.Level~Financial.Status,data=Student_data, col="tan2", 
           main = "Effect of Financial Status on the Stress level of a student", 
           xlab = "Stress Level", y = "Financial Status")

