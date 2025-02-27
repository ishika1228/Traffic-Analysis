Traffic Volume Prediction using Machine Learning

📌 Project Overview

This project aims to predict traffic volume using machine learning techniques. It utilizes a dataset containing historical traffic data with various environmental and weather-related factors. The model is designed to assist in traffic management and planning by forecasting congestion levels.

📂 Dataset

The dataset used is Metro Interstate Traffic Volume, which includes features like:

Temperature (temp)

Rainfall (rain_1h)

Snowfall (snow_1h)

Cloud Cover (clouds_all)

Weather Conditions (weather_main)

Traffic Volume (traffic_volume - Target variable)

🛠️ Technologies Used

Python

TensorFlow & Keras (Deep Learning)

Pandas & NumPy (Data Processing)

Scikit-learn (Preprocessing & Scaling)

Matplotlib & Seaborn (Data Visualization)

📌 Model Architecture

The model follows a deep learning approach using a Dense Neural Network (DNN):

Feature extraction using Lambda layer

Hidden Layer: Dense (512 neurons, ReLU activation)

Output Layer: Dense (1 neuron, Linear activation)
📊 Data Preprocessing Steps

Load dataset: Read Metro_Interstate_Traffic_Volume.csv

Handle missing values: Drop any NaN values

One-hot encode categorical variables: weather_main

Normalize numerical features: Using MinMaxScaler

Split dataset: Train (70%), Validation (20%), Test (10%)

📈 Model Training

The model is trained using Mean Squared Error (MSE) as the loss function and Adam optimizer. Early stopping is applied to prevent overfitting.

🔍 Results & Visualization

Loss & Accuracy Plots

Traffic volume trend analysis

Predictions vs Actual values

📝 Future Enhancements

Improve model performance with advanced architectures (LSTMs, CNNs)

Integrate real-time traffic API data

Deploy as a web application

🤝 Contributing

Pull requests are welcome! Feel free to submit issues and feature requests.

📜 License

This project is licensed under the MIT License.
