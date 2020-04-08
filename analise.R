library(tidyverse) # usada
library(nnfor) # usada
library(rnn) #rnn
library(MLmetrics)

# Read files with time series
ak47 = as_tibble(read.csv("./ak47_diario.csv", sep = ","))

# Looking at the data
glimpse(ak47)

# Eliminating unnecessary columns
ak47 = ak47[,c(2,3)]

# Looking at the data again
glimpse(ak47)

# Renaming the columns
names(ak47) = c("date", "price")

# Plotting the data
ak47 %>% ggplot(aes(x = 1:nrow(ak47), y = price)) +
  geom_line(col = "red")+
  labs(x = "Period",y = "Price")

# Preparing the database for model training

    # Number of past periods to be observed Y = F(X1, X2, X3)
    periodos_anteriores = 3
    
    # Dataframe preparation
    n_col = ncol(ak47)
    
    for (i in 1:periodos_anteriores) {
      for (j in 1:nrow(ak47)) {
        if (j - periodos_anteriores <= 0) {
          
        } else {
          ak47[j, n_col + i] = ak47[j - i, 2]
        }
        
      }
    }  
# Looking at the new dataframe
print(ak47)

# Preparing training data
ak47_rnn = ak47[4:(nrow(ak47)), -1]
names(ak47_rnn) = c("y", "x_1", "x_2", "x_3") 

# Normalizing the data
minmax_price = linscale(ak47_rnn$y, minmax = list(mn = 0, mx =1))
minmaax_x1 = linscale(ak47_rnn$x_1, minmax = list(mn = 0, mx =1))
minmaax_x2 = linscale(ak47_rnn$x_2, minmax = list(mn = 0, mx =1))
minmaax_x3 = linscale(ak47_rnn$x_3, minmax = list(mn = 0, mx =1))

# Transforming data into the right structure
Y <- array(minmax_price$x[1:(nrow(ak47_rnn) - 500)],
           dim=c(nrow(ak47_rnn)-500, 1))

X <- array(c(minmaax_x1$x[1:(nrow(ak47_rnn) - 500)],
             minmaax_x2$x[1:(nrow(ak47_rnn) - 500)],
             minmaax_x3$x[1:(nrow(ak47_rnn) - 500)]),
           dim=c(nrow(ak47_rnn)-500,1, 3))

# Training the recurrent neural network
rnn_ak47 = trainr(Y , X,
                 learningrate = 0.5,
                 hidden_dim = 5,
                 numepochs = 100)


# plotting the error
data.frame(Epoch = 5:length(attr(rnn_ak47, "error")),
           Error = attr(rnn_ak47, "error")[5:100]) %>%
  ggplot(aes(x = Epoch,y= Error))+
  geom_line(col = "red")+
  scale_x_continuous(breaks = seq(0, 100,10),
                   labels = seq(0, 100, 10))

# Making predictions

    # Storing the last 10 periods of the original series
    entrada_h = array(c(minmaax_x1$x[(nrow(ak47_rnn) - 500):nrow(ak47_rnn)],
                minmaax_x2$x[(nrow(ak47_rnn) - 500):nrow(ak47_rnn)],
                minmaax_x3$x[(nrow(ak47_rnn) - 500):nrow(ak47_rnn)]),
                dim=c(nrow(ak47_rnn)-(nrow(ak47_rnn) - 500),1, 3))

    # Making predictions
    pred_new =as.data.frame(predictr(rnn_ak47, entrada_h))
    pred_old =as.data.frame(predictr(rnn_ak47, X))
    # Plotting the comparison between the real vs predator

# Saving the model
saveRDS(rnn_ak47, "./rnn_elv.Rdata")

# Returning to the original scale
prev_desnorm = linscale(pred_old$V1,
                        minmax = minmax_price$minmax,
                        rev = TRUE)

prev_desnorm_new = linscale(pred_new$V1,
                        minmax = minmax_price$minmax,
                        rev = TRUE)
# Calcular os erros
MAPE(prev_desnorm$x, ak47_rnn$y[1:1575])

# Final graph
finalresult = data.frame(real = ak47_rnn$y[1:1575],
                         pred = prev_desnorm$x)

# transforming data into tidy format
finalresult$id = 1:nrow(finalresult)
finalresult %>% gather(var, value, -id) -> finalresult
finalresult = rbind(finalresult, data.frame(id = 1576:(1576+length(prev_desnorm_new$x)-1),
                              var = rep("new", length(prev_desnorm_new$x)),
                              value = prev_desnorm_new$x))

# Plotting the final results
finalresult %>% ggplot(aes(x = id,
                           y = value, colour = var))+
  geom_line(na.rm = T)+
  scale_colour_manual(values = c("Red", "Blue", "gray10"))
