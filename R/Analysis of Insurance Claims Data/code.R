library(tidyverse)
library(ggplot2)
library(RColorBrewer)
library(tidyr)
library(ggplot2)
library(lattice)
library(leaps)
library(psych)
library(AER) 
library(caret)
library(ggpubr)
library(dplyr)

claims_data <- readRDS(url('https://gmubusinessanalytics.netlify.app/data/claims_df.rds'))

claims_data

str(claims_data)

# Viewing the first few records in the dataframe 
head(claims_data)

# checking for null values in the dataset
sum(is.na(claims_data))

# 1.Is there a correlation between the total number of claims and the monthly premium, 
# indicating that the premium could being increased after every claim?

#---------------line and point plot---------

ggplot(data=claims_data, aes(x=total_claims, y=monthly_premium, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(colour="dark green") +
  ggtitle("Correlation between Total Claims and Monthly Premiums of the customers") +
  xlab("Total Claims") + ylab("Monthly Premium")

#----------density plot-----------

ggplot(claims_data, aes(x = monthly_premium)) +
  geom_density(color="black", fill="rosybrown1") +
  labs(title = "Correlation between Total Claims and Monthly Premiums of the customers") +
  facet_wrap(~total_claims)

#--------------------Summary Table-----------

claims_data %>% group_by(total_claims) %>% 
  summarise(count = n(),
            min_premium = min(monthly_premium),
            avg_premium = mean(monthly_premium),
            max_premium = max(monthly_premium),
            sd_premium = sd(monthly_premium))

# 2.Does a person being married makes them more likely to be on a Premium coverage?

#-------------Bar Plot------------------

ggplot(claims_data, aes(x=marital_status)) +
  geom_bar(fill = "turquoise", col = "black") +
  facet_wrap(~ coverage) +
  labs(title = "Effect of Marital Status on the type of coverage" ,
       x = "Marital Status",
       y = "Number of Policies") +
  geom_text(aes(label=..count..),stat='count',position=position_dodge(0.9),vjust=-0.2)


ggplot(claims_data, aes(x=marital_status)) +
  geom_bar(fill = "cornsilk", col = "black") +
  facet_wrap(~ policy) +
  labs(title = "Effect of Marital Status on the type of policy" ,
       x = "Marital Status",
       y = "Number of Policies") +
  geom_text(aes(label=..count..),stat='count',position=position_dodge(0.9),vjust=-0.2)

#-------------Mosaic Plot--------------

mosaicplot(marital_status~coverage,data=claims_data, col="slategray4", 
           main = "Effect of Marital Status on the type of coverage", 
           xlab = "Marital Status", y = "Number of Policies")

# 3.Is there a positive correlation between the type of vehicle that a person owns and their
# value to the company?

#-----------Box Plot-------------

boxplot( customer_lifetime_value~vehicle_class,
         data=claims_data,
         main="Relation between the Customer's vehicle class and their total lifetime value",
         xlab="State",
         ylab="Total Claim amount",
         col="lemonchiffon3",
         border="black"
)

#--------------------Summary Table-----------

claims_data %>% group_by(vehicle_class) %>% 
  summarise(count = n(),
            min_lifetime_value = min(customer_lifetime_value),
            avg_lifetime_value = mean(customer_lifetime_value),
            max_lifetime_value = max(customer_lifetime_value),
            sd_lifetime_value = sd(customer_lifetime_value))

#----------Heat Map-------------

ggplot(claims_data, aes(vehicle_class, policy)) +    
  geom_tile(aes(fill = customer_lifetime_value))+        
  labs(title = "Relation between the vehicle type of a customer and their policy type") +
  # geom_text(aes(label = total_claims_amount), size = 3) +
  scale_fill_gradient(low = "white", high = "purple4")


ggplot(claims_data, aes(vehicle_class, coverage)) +    
  geom_tile(aes(fill = customer_lifetime_value))+ 
  labs(title = "Relation between the vehicle type of a customer and their coverage type") +
  # geom_text(aes(label = total_claims_amount), size = 3) +
  scale_fill_gradient(low = "white", high = "royalblue4")

# 4.Does a person being employed have an impact on them claiming their policies less frequently, 
# thus making them more profitable?

#-----------density plot---------------

ggplot(claims_data, aes(x = total_claims_amount)) +
  geom_density(color="darkblue", fill="lightblue") +
  labs(title = "Relation between a customer's employment status and their claims") +
  facet_wrap(~employment_status)

#----------Heat Map-------------

ggplot(claims_data, aes(employment_status, total_claims)) +    
  geom_tile(aes(fill = total_claims_amount))+ 
  labs(title = "Relation between a customer's employment status and their claims") +
  #geom_text(aes(label = total_claims_amount), size = 3) +
  scale_fill_gradient(low = "white", high = "dark green")

#--------------Summary Table------------

claims_data %>% group_by(employment_status) %>% 
  summarise(count = n(),
            min_claims = min(total_claims),
            avg_claims = mean(total_claims),
            max_claims = max(total_claims),
            sd_claims = sd(total_claims))

# 5.Can the customers from a particular state be called more profitable to the company,
# based on their premiums and claims?

#------------line and point plot-----------

ggplot(data=claims_data, aes(x=customer_state, y=customer_lifetime_value, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(col="midnightblue") +
  ggtitle("Relation between the Customer's state and their total claim amount") +
  xlab("State") + ylab("Total Claim amount")

#--------------boxplot----------------------

boxplot( monthly_premium~customer_state,
         data=claims_data,
         main="Relation between the Customer's state and their total monthly premium",
         xlab="State",
         ylab="Monthly Premium",
         col="lightseagreen",
         border="black",
         ylim = c(50, 200)
)

#--------------------Summary Table-----------

claims_data %>% group_by(customer_state) %>% 
  summarise(count = n(),
            min_lifetime_value = min(customer_lifetime_value),
            avg_lifetime_value = mean(customer_lifetime_value),
            max_lifetime_value = max(customer_lifetime_value),
            sd_lifetime_value = sd(customer_lifetime_value))

# 6. Are certain coverage types more profitable, due to having a lower number of claims?

#--------------------Summary Table-----------

claims_data %>% group_by(coverage) %>% 
  summarise(count = n(),
            min_claims = min(total_claims),
            avg_claims = mean(total_claims),
            max_claims = max(total_claims),
            sd_claims = sd(total_claims))

#-------------boxplot---------------

boxplot( total_claims~coverage,
         data=claims_data,
         main="Relation between the policy type and total number of claims",
         xlab="Policy",
         ylab="Total Claims",
         col="navajowhite",
         border="black"
)

# 7.Does lower education levels of the customers imply that they are not as profitable, 
# referring to their total claim amount and the total number of their claims?

#----------density plot-----------

ggplot(claims_data, aes(x = total_claims)) +
  geom_density(color="black", fill="azure") +
  labs(title = "Relation between the highest education of a customer and their number of claims") +
  facet_wrap(~highest_education)


#----------Heat Map-------------

ggplot(claims_data, aes(highest_education, total_claims)) +    
  geom_tile(aes(fill = total_claims_amount))+         
  labs(title = "Relation between the highest education of a customer and their number of claims") +
  # geom_text(aes(label = total_claims_amount), size = 3) +
  scale_fill_gradient(low = "white", high = "indianred4")

#--------------------Summary Table-----------

claims_data %>% group_by(highest_education) %>% 
  summarise(count = n(),
            min_claims = min(total_claims),
            avg_claims = mean(total_claims),
            max_claims = max(total_claims),
            sd_claims = sd(total_claims))

#--------------------Summary Table-----------

claims_data %>% group_by(highest_education) %>% 
  summarise(count = n(),
            min_claim_amount = min(total_claims_amount),
            avg_claim_amount = mean(total_claims_amount),
            max_claim_amount = max(total_claims_amount),
            sd_claim_amount = sd(total_claims_amount))

# 8.Is there a very large profit margin between the customers with different types of vehicle classes? 
# Compare the top two vehicle classes with the rest to determine this.

#--------------------Summary Table-----------

claims_data %>% group_by(vehicle_class) %>% 
  summarise(count = n(),
            min_claim_amount = min(total_claims_amount),
            avg_claim_amount = mean(total_claims_amount),
            max_claim_amount = max(total_claims_amount),
            sd_claim_amount = sd(total_claims_amount))

#---------------histogram---------------

ggplot(claims_data, aes(total_claims_amount))+ 
  labs(title = "Histogram of total claim amount of a customer with respect to Vehicle type") +
  geom_histogram(bins=30, fill = "coral1", col = "black")+facet_grid(.~vehicle_class)

#--------------------Summary Table-----------

claims_data %>% group_by(vehicle_class) %>% 
  summarise(count = n(),
            min_premium = min(monthly_premium),
            avg_premium = mean(monthly_premium),
            max_premium = max(monthly_premium),
            sd_premium = sd(monthly_premium))

#---------------histogram---------------

ggplot(claims_data, aes(monthly_premium))+ 
  labs(title = "Histogram of monthly premium with respect to Vehicle type") +
  geom_histogram(bins=30, fill = "darkkhaki", col = "black")+facet_grid(.~vehicle_class)

#----------------histogram--------------

ggplot(claims_data, aes(months_policy_active))+
  labs(title = "Histogram of the active time of policies with respect to Vehicle type") +
  geom_histogram(bins=30, fill = "darkolivegreen1", col = "black")+facet_grid(.~vehicle_class)

# 9.Do the customers with higher income keep their policies active for longer without less claims, 
# thus making them long-term and preferable customers for the insurance company?


#--------------------Summary Table-----------

claims_data %>% group_by(total_claims) %>% 
  summarise(count = n(),
            min_income = min(income),
            avg_income = mean(income),
            max_income = max(income),
            sd_income = sd(income))

#-----------Box Plot-------------

boxplot( income~total_claims,
         data=claims_data,
         main="Relation between the Customer's Income and their total number of claims",
         xlab="Total Claims",
         ylab="Income",
         col="plum3",
         border="black"
)


