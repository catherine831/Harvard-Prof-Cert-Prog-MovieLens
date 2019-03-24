#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Install and Load Packages
install.packages("plotly")
library(dplyr)
library(ggplot2)
library(caret)
library(tidyr)
library(plotly)

# Question 1: How many rows and columns are there in the edx dataset?
dim(edx)
rating_no <- nrow(edx)

# Question 2: How many zeros were given as ratings in the edx dataset? 
# & How many threes were given as ratings in the edx dataset?
edx %>% filter(rating == 0) %>% tally()
edx %>% filter(rating == 3) %>% tally()

# Question 3: How many different movies are in the edx dataset?
n_distinct(edx$movieId)
movie_no<-n_distinct(edx$movieId)

# Question 4: How many different users are in the edx dataset?
n_distinct(edx$userId)
user_no<-n_distinct(edx$userId)

# Question 5: How many movie ratings are in each of the following genres in the edx dataset?
# Drama / Comedy / Thriller / Romance
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

# Question 6: Which movie has the greatest number of ratings?
edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

# Question 7: What are the five most given ratings in order from most to least?
edx %>% group_by(rating) %>% summarize(count = n()) %>% top_n(5) %>%
  arrange(desc(count)) 

# Question 8a: True or False: In general, half star ratings are less common than whole star ratings (e.g., there are fewer ratings of 3.5 than there are ratings of 3 or 4, etc.); OR 
edx %>% group_by(rating) %>% summarize(count = n())

# Question 8b
edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()


#############################################################
# Create a train set and test set from edx dataset
#############################################################

# test set will be 20% of edx data
set.seed(1)
edx_test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-edx_test_index,]
test_set <- edx[edx_test_index,]

# To make sure excluding users and movies in the test set that do not appear in the training set
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")


#############################################################
# Define Residual Mean Squared Error (RMSE) 
#############################################################

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


#############################################################
# Model: Simplest model without any effects (Average Rating)
#############################################################

# Average Rating
mu_hat <- mean(train_set$rating)
mu_hat

# Prediction of Test Set
naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse

# Create a Results Table
rmse_results <- data_frame(method = "Just the Average", RMSE = naive_rmse)

rmse_results


#############################################################
# Model: Movie Effects
#############################################################

# Average Rating
mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Distribution of Rating
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

# Prediction of Test Set
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

movie_rmse <- RMSE(predicted_ratings, test_set$rating)

# Add movie effect approach result to the Results Table
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",  
                                     RMSE = movie_rmse))
rmse_results


#############################################################
# Model: Movie Effects and User Effects
#############################################################

# Distribution of Average Rating for users u rated over 100 movies
train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

# Average Rating
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Prediction of Validatation Set (Test Set)
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

movie_user_rmse <- RMSE(predicted_ratings, test_set$rating)

# Add user effect approach result to the Results Table
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = movie_user_rmse))

rmse_results %>% knitr::kable()


#############################################################
# Model: Regularization Movie & User Effect Model
#############################################################

# Use cross-validation to pick a λ (lambdas)
lambdas <- seq(0, 10, 0.25)

reg_rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, reg_rmses)  

# The lambda for the best RMSE
lambda <- lambdas[which.min(reg_rmses)]
lambda

# Add Regularization with movive effect and  user effect approach result to the Results Table
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = min(reg_rmses)))
rmse_results %>% knitr::kable()


##################################################################
# RMSE for Validation Set by Regularized Movie & User Effect Model
##################################################################

# Use cross-validation to pick a λ (lambdas)
val_lambdas <- seq(0, 10, 0.25)

validation_rmses <- sapply(val_lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})

qplot(lambdas, validation_rmses)  

# The lambda for the best RMSE
val_lambda <- val_lambdas[which.min(validation_rmses)]
val_lambda

# RMSE for Validation Set
min(validation_rmses)
