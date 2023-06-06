library(tidyverse)
library(ggplot2)
library(dplyr)
library(RColorBrewer)
library(tidyr)
library(expss)
library(lattice)
library(leaps)
library(psych)
library(caTools)
library(caret)
library(Hmisc)

development_data = read.csv("C:/Users/Public/Cleaned_Development_data.csv")

head(development_data)

# 1. How many number of projects are there per Project type?

table(development_data$ProjType)

# 5. Identify the range for the Square Footage (area) of projects and Square Footage (area) per category.
# Also identify the summary statistics for the Square Footage (area) of projects.

SqFootage_range = range(development_data$SqFootage, na.rm = TRUE)

SqFootage_range

boxplot(development_data$SqFootage, horizontal = TRUE, 
        main = "Square Footage of the Projects",
        xlab = "Square Footage",
        ylab = "",
        col = "steelblue",
        border = "black",
        ylim = c(0 , 6e+04))
x = development_data$SqFootage
text(x = fivenum(x),labels = fivenum(x),y = 1.25)

boxplot(SqFootage~Category , 
        data = development_data ,
        main = "Square Footage per Category" ,
        xlab ="Project Category" ,
        ylab ="Square Footage per category",
        ylim = c(0 , 150000))

SqFootage_summary = summary(development_data$SqFootage, na.rm = TRUE)

SqFootage_summary

# 8.	How many projects have been actioned upon for more than a year, since the initial application date?

development_data$duration <- development_data$LastYear - development_data$CreYear

names(development_data)

prop.table(table(development_data$duration))

sum(development_data$duration == 0)

sum(development_data$duration == 1)

sum(development_data$duration == 2)

sum(development_data$duration == 3)

sum(development_data$duration == 4)

# Visual representation of the individual counts of the projects, grouped by the number of active years:

ggplot(development_data, aes(x = duration)) +
  geom_bar(fill = "lavender", color = "black", width = 0.5) +
  geom_text(aes(label = ..count..), stat = "count", vjust = -0.7, colour = "black") +
  scale_fill_brewer(palette = "Accent")

# 9.What proportion of projects have received the required approval of construction from the 
# appropriate Boards/ Committees?

# BOZA

ggplot(development_data, aes(x = BOZA)) +
  geom_bar(fill = "maroon", color = "black", width = 0.1) +
  geom_text(aes(label = ..count..), stat = "count", vjust = -0.7, colour = "black") +
  ggtitle("Proportion of Applications actioned upon by BOZA")

# PC

ggplot(development_data, aes(x = PC)) +
  geom_bar(fill = "orange", color = "black", width = 0.4) +
  geom_text(aes(label = ..count..), stat = "count", vjust = -0.7, colour = "black") +
  scale_fill_brewer(palette = "Accent") +
  ggtitle("Proportion of Applications actioned upon by PC")

# DRC

ggplot(development_data, aes(x = DRC)) +
  geom_bar(fill = "cyan", color = "black", width = 0.4) +
  geom_text(aes(label = ..count..), stat = "count", vjust = -0.7, colour = "black") +
  scale_fill_brewer(palette = "Accent") +
  ggtitle("Proportion of Applications actioned upon by DRC")

# LDT

ggplot(development_data, aes(x = LDT)) +
  geom_bar(fill = "purple", color = "black", width = 0.4) +
  geom_text(aes(label = ..count..), stat = "count", vjust = -0.7, colour = "black") +
  scale_fill_brewer(palette = "Accent") +
  ggtitle("Proportion of Applications actioned upon by LDT")

  # Regression model to determine Project Type from the number of dwelling units: 

development_data$ProjType <- as.factor(development_data$ProjType)

set.seed(1000)

split <- sample.split(development_data, SplitRatio = 0.8)

train <- subset(development_data, split == "TRUE")
test <- subset(development_data, split == "FALSE")

logistc_model <- glm(ProjType ~ Units, data = train, family = 'binomial')
summary(logistc_model)

par(mfrow=c(2,2))
plot(logistc_model)
par(mfrow=c(1,1))




