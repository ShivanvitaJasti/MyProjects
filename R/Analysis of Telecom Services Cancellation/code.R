## Add R libraries here

library(tidyverse)
library(tidymodels)
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
library(ggplot2)
library(lattice)
library(leaps)
library(psych)
library(caTools)
library(caret)
library(Hmisc)
library(ranger)
library(rlang)
library(cli)
library(vctrs)
library(glue)
library(tibble)
library(tidyselect)

# Load the dataset
telecom_data <- readRDS(url('https://gmubusinessanalytics.netlify.app/data/telecom_df.rds'))

head(telecom_data)

str(telecom_data)

sum(is.na(telecom_data))

write.csv(telecom_data,"C:\\Users\\Public\\df.csv", row.names = FALSE)

# 1. Does the person having more average call and international call minutes make them more likely to retain their cell service?

#-----------Box Plot-------------

boxplot( avg_call_mins~canceled_service,
         data=telecom_data,
         main="Relation between the Customer's average call minutes and service cancellation",
         xlab="Canceled Service",
         ylab="Average call minutes",
         col="lemonchiffon3",
         border="black"
)

# people with more call minutes cancel their service more, probably to get upgraded to a better plan
# that suits them

#--------------------Summary Table-----------

telecom_data %>% group_by(canceled_service) %>% 
  summarise(count = n(),
            min_call_mins = min(avg_call_mins),
            avg_call_mins = mean(avg_call_mins),
            max_call_mins = max(avg_call_mins),
            sd_call_mins = sd(avg_call_mins))

#-----------Box Plot-------------

boxplot( avg_intl_mins~canceled_service,
         data=telecom_data,
         main="Relation between the Customer's average international call minutes \n and service cancellation",
         xlab="Canceled Service",
         ylab="Average international call minutes",
         col="lemonchiffon3",
         border="black"
)

# people with more international call minutes cancel their service less, probably because there are not very
# many plans that suit international call requirements

#--------------------Summary Table-----------

telecom_data %>% group_by(canceled_service) %>% 
  summarise(count = n(),
            min_intl_mins = min(avg_intl_mins),
            avg_intl_mins = mean(avg_intl_mins),
            max_intl_mins = max(avg_intl_mins),
            sd_intl_mins = sd(avg_intl_mins))

# 2. Is it more likely for people to cancel their service early on, and the people who do not cancel their service to stay on the service for longer periods?

#-----------Box Plot-------------

boxplot( months_with_company~canceled_service,
         data=telecom_data,
         main="Relation between the Customer's number of months \n with the company and service cancellation",
         xlab="Canceled Service",
         ylab="Number of months with the company",
         col="lemonchiffon3",
         border="black"
)

#-----------density plot---------------

ggplot(telecom_data, aes(x = months_with_company)) +
  geom_density(color="darkblue", fill="lightblue") +
  labs(title = "Relation between the Customer's number of months with the \n company and service cancellation") +
  facet_wrap(~canceled_service)

#--------------------Summary Table-----------

telecom_data %>% group_by(canceled_service) %>% 
  summarise(count = n(),
            min_months = min(months_with_company),
            avg_months = mean(months_with_company),
            max_months = max(months_with_company),
            sd_months = sd(months_with_company))


# there is a significant gap between the people who cancel their service with respect to the duration
# for which they have stayed with the company
# This shows that more importance must be given to the customers who have newly enrolled with the company
# to ensure that their needs are being adequately met, in order to make them long-term customers
# who are less likely to cancel their plans, according to the visualizations displayed above

# 3. Does the type of internet service a person is enrolled in have an impact on them retaining their cell service?

#-------------Mosaic Plot--------------

mosaicplot(internet_service~canceled_service,data=telecom_data, col="slategray4", 
           main = "Relation between the Customer's internet service type and service cancellation", 
           xlab = "Internet Service", y = "Service cancellation ")

# 4. Does a person having online security and device protection enabled have a positive effect on them not canceling their cell service?


#----------Heat Map-------------

ggplot(telecom_data, aes(canceled_service, online_security)) +    
  geom_tile(aes(fill = months_with_company))+        
  labs(title = "Relation between the Customer's online security option and service cancellation") +
  # geom_text(aes(label = total_claims_amount), size = 3) +
  scale_fill_gradient(low = "white", high = "purple4")

# this map shows that the customers with online security option enabled are more likely to cancel their
# plan and stay with the company for lesser durations
# this could be attributed to the online security option being pricey, thereby forcing the users
# to cancel

#----------Heat Map-------------

ggplot(telecom_data, aes(canceled_service, device_protection)) +    
  geom_tile(aes(fill = months_with_company))+        
  labs(title = "Relation between the Customer's device protection option and service cancellation") +
  # geom_text(aes(label = total_claims_amount), size = 3) +
  scale_fill_gradient(low = "white", high = "royalblue4")

# this heatmap demonstrates that customers with device protection option enabled are far less likely to 
# cancel their plans and stay with the company for longer terms.

# 5. Does the type of contract a person is registered with have an impact on them staying with the company for longer?

#---------------histogram---------------

ggplot(telecom_data, aes(months_with_company))+ 
  labs(title = "Histogram of duration a customer stays with the company with respect to their contract type") +
  geom_histogram(bins=30, fill = "coral1", col = "black")+facet_grid(.~contract)

#--------------------Summary Table-----------

telecom_data %>% group_by(contract) %>% 
  summarise(count = n(),
            min_months = min(months_with_company),
            avg_months = mean(months_with_company),
            max_months = max(months_with_company),
            sd_months = sd(months_with_company))

# It can be clearly illustrated from the visualization that majority of the customers with a 
# month-to-month plan cancel their plans with the company after a very short period of time. 
# A sharp contrast with this phenomenon can be observed with the customers who are enrolled in 
# one year and two year plans

# 6. Is a person with lesser monthly charges more likely to not cancel their service?

#-----------Box Plot-------------

boxplot( monthly_charges~canceled_service,
         data=telecom_data,
         main="Relation between the Customer's number of months with the \n company and service cancellation",
         xlab="Canceled Service",
         ylab="Number of months with the company",
         col="rosybrown1",
         border="black"
)

#--------------------Summary Table-----------

telecom_data %>% group_by(canceled_service) %>% 
  summarise(count = n(), 
            min_charges = min(monthly_charges),
            avg_charges = mean(monthly_charges),
            max_charges = max(monthly_charges),
            sd_charges = sd(monthly_charges))

# there is no significant difference between the customers who cancel their service and those who do 
# not, with respect to their monthly charges. But the average monthly charges of the customers who do
# not cancel their service is slightly lesser than that of those who did


#---------------------MACHINE LEARNING-------------------

#--------Splitting the data-------

set.seed(123)
split1<- sample(c(rep(0, 0.8 * nrow(telecom_data)), rep(1, 0.2 * nrow(telecom_data))))
train <- telecom_data[split1 == 0, ] 
test <- telecom_data[split1== 1, ] 

#--------------Feature Engineering with recipes-----------

my_recipe <- 
  recipe(canceled_service ~ ., data = train) %>% 
  step_center(all_numeric_predictors()) %>% 
  step_scale(all_numeric_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>%
  step_corr(all_numeric(), -all_outcomes(), threshold = 0.2) %>%
  step_dummy(all_nominal(), -all_outcomes())

# processed_train <- my_recipe %>% 
#   prep() %>% 
#   bake(new_data = train)
# 
# processed_test <- my_recipe %>% 
#   prep() %>% 
#   bake(new_data = test)
# 
# 
# processed_train
# processed_test

#-----------LOGISTIC REGRESSION---------------

#defining model specifics

LR_specifics <- 
  logistic_reg() %>% 
  set_engine("glm") %>%
  set_mode("classification")

# creating workflow

LR_workflow <- 
  workflow() %>% 
  add_model(LR_specifics) %>% 
  add_recipe(my_recipe)

# fitting the model to training data

fit_model_LR <- LR_workflow %>% fit(data = train)

# evaluating the model with ROC curve

probs <- fit_model_LR %>%
  predict(test, type = "prob") %>%  bind_cols(test)

probs%>%
  roc_curve(canceled_service, .pred_yes) %>% autoplot()

# evaluating the metrics of the model

predict(fit_model_LR, test, type = "prob") %>%
  bind_cols(predict(fit_model_LR, test)) %>%
  bind_cols(select(test, canceled_service)) %>%
  metrics(canceled_service, .pred_yes, estimate = .pred_class)

# confusion matrix of the model

LR_res <- test %>% select(canceled_service) %>%
  bind_cols(fit_model_LR %>% predict(new_data = test)) %>% 
  bind_cols(fit_model_LR %>% predict(new_data = test, type = "prob"))

conf_mat(LR_res,
         truth = canceled_service,
         estimate = .pred_class)

conf_mat(LR_res,
         truth = canceled_service,
         estimate = .pred_class)%>% 
  autoplot(type = 'heatmap')


#-----------------------RANDOM FORESTS------------------

#defining model specifics

RF_specifics <- 
  rand_forest() %>% 
  set_engine("ranger") %>%
  set_mode("classification")

# creating workflow

RF_workflow <- 
  workflow() %>% 
  add_model(RF_specifics) %>% 
  add_recipe(my_recipe)

# fitting the model to training data

fit_model_RF <- RF_workflow %>% fit(data = train)

# evaluating the model with ROC curve

probs <- fit_model_RF %>%
  predict(test, type = "prob") %>%  bind_cols(test)

probs%>%
  roc_curve(canceled_service, .pred_yes) %>% autoplot()

# evaluating the metrics of the model

predict(fit_model_RF, test, type = "prob") %>%
  bind_cols(predict(fit_model_RF, test)) %>%
  bind_cols(select(test, canceled_service)) %>%
  metrics(canceled_service, .pred_yes, estimate = .pred_class)

# confusion matrix of the model

RF_res <- test %>% select(canceled_service) %>%
  bind_cols(fit_model_RF %>% predict(new_data = test)) %>% 
  bind_cols(fit_model_RF %>% predict(new_data = test, type = "prob"))

conf_mat(RF_res,
         truth = canceled_service,
         estimate = .pred_class)

conf_mat(RF_res,
         truth = canceled_service,
         estimate = .pred_class)%>% 
         autoplot(type = 'heatmap')

#----------------DECISION TREE-------------------


#defining model specifics

DT_specifics <- 
  decision_tree() %>% 
  set_engine("rpart") %>%
  set_mode("classification")

# creating workflow

DT_workflow <- 
  workflow() %>% 
  add_model(DT_specifics) %>% 
  add_recipe(my_recipe)

# fitting the model to training data

fit_model_DT <- DT_workflow %>% fit(data = train)

# evaluating the model with ROC curve

probs <- fit_model_DT %>%
  predict(test, type = "prob") %>%  bind_cols(test)

probs%>%
  roc_curve(canceled_service, .pred_yes) %>% autoplot()

# evaluating the metrics of the model

predict(fit_model_DT, test, type = "prob") %>%
  bind_cols(predict(fit_model_DT, test)) %>%
  bind_cols(select(test, canceled_service)) %>%
  metrics(canceled_service, .pred_yes, estimate = .pred_class)

# confusion matrix of the model

DT_res <- test %>% select(canceled_service) %>%
  bind_cols(fit_model_DT %>% predict(new_data = test)) %>% 
  bind_cols(fit_model_DT %>% predict(new_data = test, type = "prob"))

conf_mat(DT_res,
         truth = canceled_service,
         estimate = .pred_class)

conf_mat(DT_res,
         truth = canceled_service,
         estimate = .pred_class)%>% 
  autoplot(type = 'heatmap')


#--------------------KNN--------------------

#defining model specifics

KNN_specifics <- 
  nearest_neighbor() %>% 
  set_engine("kknn") %>%
  set_mode("classification")

# creating workflow

KNN_workflow <- 
  workflow() %>% 
  add_model(KNN_specifics) %>% 
  add_recipe(my_recipe)

# fitting the model to training data

fit_model_KNN <- KNN_workflow %>% fit(data = train)

# evaluating the model with ROC curve

probs <- fit_model_KNN %>%
  predict(test, type = "prob") %>%  bind_cols(test)

probs%>%
  roc_curve(canceled_service, .pred_yes) %>% autoplot()

# evaluating the metrics of the model

predict(fit_model_KNN, test, type = "prob") %>%
  bind_cols(predict(fit_model_KNN, test)) %>%
  bind_cols(select(test, canceled_service)) %>%
  metrics(canceled_service, .pred_yes, estimate = .pred_class)

# confusion matrix of the model

KNN_res <- test %>% select(canceled_service) %>%
  bind_cols(fit_model_KNN %>% predict(new_data = test)) %>% 
  bind_cols(fit_model_KNN %>% predict(new_data = test, type = "prob"))

conf_mat(KNN_res,
         truth = canceled_service,
         estimate = .pred_class)

conf_mat(KNN_res,
         truth = canceled_service,
         estimate = .pred_class)%>% 
  autoplot(type = 'heatmap')











