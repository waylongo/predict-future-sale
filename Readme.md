# Predict Future Sales competition in Kaggle

> Wenlong Wu, University of Missouri

## Background and motivation

This project idea comes from one of the competitions in Kaggle, which is the world’s largest community of data scientists and machine learners. In this project, we will work with a challenging time-series dataset consisting of daily sales data, kindly provided by one of the largest Russian software firms - [1C Company](http://1c.ru/eng/title.htm). 

We implemented some time series techniques we have learned in class (such as ARIMA) and some more advanced algorithms (such as LSTM) on this dataset. We will discuss some specific models that might be used in below model consideration.  The goal is to predict total sales for every product and store in the next month. 

## Data description

We are provided with daily historical sales data. The task is to forecast the total amount of products sold in very for the test set. Note that this dataset is a real-world dataset. It may have some missing or unreasonable values but we think dealing with dirty data and creating a robust model is also part of the fun in this project. 

The data overview is shown in Table 1. 

|                 | **Train**         | **Test**    |
| --------------- | ----------------- | ----------- |
| **Data**        | Item x shop x day | Item x shop |
| **Period**      | 34 months         | 1 month     |
| **Aggregation** | daily             | monthly     |

Table 1. Data overview

File description (from Kaggle): the website (https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data) provides several csv files for different data.

·         **sales_train.csv** - the training set. Daily historical data from January 2013 to October 2015.

·         **test.csv** - the test set. You need to forecast the sales for these shops and products for November 2015.

·         **items.csv** - supplemental information about the items/products.

·         **item_categories.csv** - supplemental information about the items categories.

·         **shops.csv** - supplemental information about the shops.

Data fields (from Kaggle):

·         **shop_id** - unique identifier of a shop

·         **item_id** - unique identifier of a product

·         **item_category_id** - unique identifier of item category

·         **item_cnt_day** - number of products sold. 

·         **item_price** - current price of an item

·         **date** - date in format dd/mm/yyyy

·         **date_block_num -** a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33)

# Exploratory Data Analysis (EDA)

The *item_cnt_day* (number of products sold in training set) is provided in the training set as the response variable, we are looking for an adequate model to predict the monthly amount of this measure in the test set. 

  For convenient analysis, the merge work has been done. In the training dataset after merging, it contains 60 shops with 22171 different items, which from 84 categories. 

![training data overview](https://github.com/waylongo/predict-future-sale/blob/master/figures/overview.png)								      		

​								Figure 1: training data overview

Because we are predicting a monthly amount of items sold, the aggregated monthly data has been introduced as well. After that, we did the data visualization of whole company sale in these 34 months. In Figure 2, we can tell there is seasonality and decreasing trend exist in this data set.  To estimate the trend component and seasonal component, we tried to use the “*decompose ()*” function in R. This function estimates the trend, seasonal, and irregular components of a time series that can be described using an additive model.

   ![](https://github.com/waylongo/predict-future-sale/blob/master/figures/month_sales.png)

​							Figure 2: Monthly sale for the whole company

  Because the final goal is to predict the monthly sale of each product in every shop. Then, we looked at the total sale in every shop, Figure 3 shows that some shops have a higher total sale, and some of them even have a small amount of total sale. So the shop factor should be considered as an important feature in the model. Also, the sale under each item category has been checked as well, it shows a significant impact on the monthly sale. For some item categories, they even have no sale as all for total 34 months 

We find each category has different number of items. But this doesn’t give us much insights of what we really want to predict. So next we will explore “*item-cnt-day*” this response feature, shown in Figure 3.

​    ![](https://github.com/waylongo/predict-future-sale/blob/master/figures/sale_1.png)

​					Figure 3. Number of items sold in each month in different shops

  Clearly, there are some seasonal patterns in this data. The number of items sold in the end of each year is larger than that of other months, which may be caused by holiday season. We also have some item in some periods with zero even negative sales. Some of this could be due to seasonal or returned products. However, this company is located in Russia, all the item categories and names were wrote in Russian, which makes us cannot clear catch the holiday season and item types. 

# Implementation and Results

  The first step was to use Random Forest for variable selection. It might give us some idea of which variables were valuable. However, because of time limitation, in this final project, we have tried three different models on this dataset: ETS Model (Time Series Implementation), LSTM and XGBoost. The ETS Model is a traditional time series analysis model. The key in this model is to determine the best (A, A, A) order. The LSTM model is a recurrent neural network model. It fits well on large scale of data but needs much training. The above two methods are mainly explored in this project. Furthermore, the XGBoost is a new gradient boosting tree method that is introduced in 2016. It works extremely well in the competition, so the XGBoost is also tried in this project. 

IV.A. ETS Model

  For this grouped time series analysis based on shop and item_id, ETS models with seasonality were tested and selected. Since all the levels based on shop and item id are independent, the ETS models were created as matrix to predict each level independently. For ETS model it represents:

Error: Multiplicative because the variation in the remainder shows a change in variance over time as seen in Figure 4. 

  Trend: Additive since the trend line is linear as seen in the decomposition plot above.

  Seasonality: Multiplicative as since the remainder is varying in magnitude every year. From figure here, we can clear see this dataset has obvious "seasonality" and a decreasing "Trend".

![](https://github.com/waylongo/predict-future-sale/blob/master/figures/deconpostion.png)

​								Figure 4. The decomposition plot

  One of the simple methods for generating coherent forecasts is the bottom-up approach. It involves first generating forecasts for each series at the bottom-level, and then summing these to produce forecasts for all the series in the structure. It has some advantages, simple aggregate and no information is lost due to aggregation. But lower level aggregate could cause noise.  Also if time allows we would like to try Top-down and middle-out approach as well. 

  For the monthlysale for each shop and each item, the shop_id and item_is have beem combined as shopitem_id. From the overall and sale data from each shop, there are pick points as month 11 and 23 with trends. So before ETS estimation, the data has been dcast from long to wide format for time series forecasting. The forecast interval has been set as 1, since it is only predict for the next one month sale with considering shopitem_id. Fifteen cores have been created for parallel processing in ETS forecasting.  Also I tried the hierarchy time series by using S matrices with grouped structure. 12-step-ahead bottom-up forecasts using ETS models for the check the insight of the grouped time seires. For online search, the accuracy () command is useful for evaluating the forecast accuracy across hierarchical or grouped structures. A holdout last month sale as sample to test the models. The ETS models are now tested by selecting lowest AIC = 482.967. 

The final submission of ETS has returned score 1.14840, which is around top 50% ranking in the public board.

![](https://github.com/waylongo/predict-future-sale/blob/master/figures/ets.png)

IV.B. LSTM model

The recurrent neural network has been shown great performance on time series data. Long short-term memory (LSTM) is one of the powerful techniques in the recurrent neural network. With the rapid development of modern machine learning frameworks such as TensorFlow, Keras and PyTorch, we could implement these complicated algorithms much easier than before. So we implemented LSTM model in this project.

  Since we have 34 months data and the data is in time series, we will use last month (October 2015) as validation set and use the other 33 month (January 2013 to September 2015) as training set. Then the model will be submitted to the Kaggle platform and evaluated using the test set occurred in November 2015.

  After the preprocessing of the data, we have the training and validation sets. The shape of training and validation sets are shown in below: 

X Train Shape:  (192780, 33, 2) - y Train Shape:  (192780, 1)

X Valid Shape:  (21420, 33, 2) - y Valid Shape:  (21420, 1)

  Then we built the LSTM model using the modern framework Keras.

```
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, ThresholdedReLU, MaxPooling2D, Embedding, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
K.clear_session()
model_lstm = Sequential()
model_lstm.add(LSTM(16, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
model_lstm.add(Dropout(0.5))
model_lstm.add(LSTM(32))
model_lstm.add(Dropout(0.5))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer="adam", loss='mse', metrics=["mse"])
print(model_lstm.summary())
```

The LSTM modelling staging is shown in below:

```
Layer (type)                 Output Shape              Param #   
=================================================================
lstm_1 (LSTM)                (None, 33, 16)            1216      
_________________________________________________________________
dropout_1 (Dropout)          (None, 33, 16)            0         
_________________________________________________________________
lstm_2 (LSTM)                (None, 32)                6272      
_________________________________________________________________
dropout_2 (Dropout)          (None, 32)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 33        
=================================================================
Total params: 7,521
```

After modelling, we use fit function to train the model.

```
callbacks_list=[EarlyStopping(monitor="val_loss",min_delta=.001, patience=3,mode='auto')]
hist=model_lstm.fit(X_train,y_train,validation_data=(X_valid,y_valid),callbacks=callbacks_list, **LSTM_PARAM)
pred = model_lstm.predict(test)
```

The training and validation curve is shown in Figure 5.

![](https://github.com/waylongo/predict-future-sale/blob/master/figures/train_valid_curve.png)

​								Figure 5. Training and validation curve

  I submitted the predicted set to Kaggle and got public score 1.01933, around top 40% ranking in the public board.

![](https://github.com/waylongo/predict-future-sale/blob/master/figures/lstm.png)

  Then changed the LSTM structures and tuned the hyper-parameters. The above result is the best we can get for now. This is OK but it should have lots of improvements to make. For now, we only use “item_id”, “shop_id”, “item_price” and “date” these a few features. This may not be enough to make a good prediction.

  So we created and took references of other features, such as “lag” features, “last sale” features and etc. The feature generation procedure is complex and may not in the scope of the class so it is not discussed here. 

  The training and validation curve using more features is shown in Figure 6.

![](https://github.com/waylongo/predict-future-sale/blob/master/figures/train_valid_features.png)

​						Figure 6. Training and validation curve with more features

  Then the result of LSTM with more features is 0.94356, around top 30% ranking in the public board.

![](https://github.com/waylongo/predict-future-sale/blob/master/figures/lstm_features.png)

IV.C. XGBoost model (gradient boosting tree method)

  The XGBoost model is widely used in the data science competitions. We would like to see its performance on this dataset. 

  The xgboost function is available in Python library. The hyper-parameters that produces best results are shown below.

```
from xgboost import XGBRegressor

model = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=20)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 10)

Y_test = model.predict(X_test).clip(0, 20)

```

  I submitted the XGBoost predicted result to Kaggle and got public score 0.90646, around top 15% ranking in the public board.

![](https://github.com/waylongo/predict-future-sale/blob/master/figures/xgboost.png)

## Conclusions

  After trying different models (ETS, LSTM and XGBoost) and fine tuning the hyper-parameters, the performance of each model is shown in Table II. XGBoost, as a gradient boosting tree method, achieves the best performance among three models. LSTM model takes much training time than the other two but achieves middle ranking performance. The reason behind this is because LSTM need large amount of data to train but the dataset we used in this project is relative small. ETS, as a traditional time series analysis, does not fit this dataset well as the other two. That may be due to the reason that the order in ETS model is hard to capture.

|             | **RMSE** | **Kaggle   ranking**   **in   the public board** |
| ----------- | -------- | ------------------------------------------------ |
| **ETS**     | 1.14840  | Top 50%                                          |
| **LSTM**    | 0.93456  | Top 30%                                          |
| **XGBoost** | 0.90646  | Top 15%                                          |

​						Table II. Performance of each model

## References

[1] Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2018). Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization. *J American Statistical Association*, *to appear*. https://robjhyndman.com/publications/mint

[2] Hyndman, R. J., Ahmed, R. A., Athanasopoulos, G., & Shang, H. L. (2011). Optimal combination forecasts for hierarchical time series. *Computational Statistics and Data Analysis*, *55*(9), 2579–2589.  https://robjhyndman.com/publications/hierarchical/

[3] https://www.kaggle.com/dlarionov/feature-engineering-xgboost

[4] https://www.kaggle.com/nicapotato/multivar-lstm-ts-regression-keras

