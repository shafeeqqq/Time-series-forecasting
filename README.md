# Time-series-forecasting
Time series prediction using LSTM in Keras

This was the result of a sudden curiosity with LSTM networks and its application in finance for time series prediction. It was mostly recreated from [this tutorial](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/) by Jason Brownlee and this [video](https://www.youtube.com/watch?v=ftMq5ps503w) from Siraj Raval.

On a high level, the LSTM network here takes considers a sequence of 20 (as specified by the sequence_length variable in the code) numbers and outputs a single number. This number is essentially the prediction based on the past 20 data points. . For a detailed explanation, visit the links in the references. 

### Data
The data (sp500.csv) was downloaded from this [site](https://github.com/llSourcell/How-to-Predict-Stock-Prices-Easily-Demo).

### Python Libraries
- tensorflow
- matplotlib
- numpy
- scikit-learn

### References
- https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
- https://www.youtube.com/watch?v=ftMq5ps503w

