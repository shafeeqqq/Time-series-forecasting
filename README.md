# Time-series-forecasting
Time series prediction using LSTM in Keras

This was the result of a sudden curiosity with LSTM networks and its application in finance for time series prediction. It was mostly recreated from [this tutorial](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/) by Jason Brownlee and this [video](https://www.youtube.com/watch?v=ftMq5ps503w) from Siraj Raval.

On a high level, the LSTM network here takes considers a sequence of 50 (as specified by the sequence_length variable in the code) numbers and outputs a single number. This number is essentially the "prediction" based on the past 50 data points. The figure below shows how the predictions (blue) compares the true values. For a detailed explanation, visit the links in the references. 

![seq-50](https://user-images.githubusercontent.com/32190446/51441865-402f4500-1d11-11e9-814a-2d4317c1b286.png)

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

