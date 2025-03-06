House Price Prediction

Project Overview

This project aims to predict house prices based on various features such as the number of bedrooms, number of bathrooms, square footage, distance to the city center, and house age. It utilizes multiple machine learning models, including Linear Regression, Decision Tree, Random Forest, and Gradient Boosting to analyze the dataset and compare their performance.

Dataset

The dataset contains information on house features and their respective prices. The key features used in this project are:

Number of Bedrooms

Number of Bathrooms

Square Footage

Distance to City Centre

House Age

Price (Target Variable)

Installation & Requirements

To run this project, install the following dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn

Project Structure

House_Price_Prediction/
│── dataset/house_price_data.csv
│── house_price_prediction.ipynb
│── README.md

Implementation

1. Data Preprocessing

Load the dataset using Pandas.

Handle missing values (if any).

Split the dataset into training and testing sets.

2. Model Training

Linear Regression Model

Trained on 80% of the dataset.

Evaluated using Mean Absolute Error, Root Mean Squared Error, and R-squared Score.

Decision Tree Model

Trained and tested similarly to Linear Regression.

Random Forest Model

Used as an ensemble learning method to improve accuracy.

Gradient Boosting Model

Applied to enhance prediction performance.

3. Model Evaluation & Comparison

Performance of different models is compared using:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R-squared Score (R²)

4. Prediction Example

To predict the price of a house with specific features:

new_house = np.array([[2, 2, 900, 1, 12]])  # Example Input
predicted_price = model.predict(new_house)
print(f"Predicted House Price: ₹{predicted_price[0]:,.2f} lakhs")

Results

Model

MAE

RMSE

R² Score

Linear Regression

0.510483

0.566914

0.996039

Decision Tree

5.500000

6.720615

0.443341

Random Forest

4.143333

4.948114

0.698248

Gradient Boosting

3.290774

3.780857

0.823822

Conclusion

This project demonstrates the effectiveness of different machine learning models in predicting house prices. Among them, the Linear Regression model performs best with the highest R² score, indicating a strong predictive capability.

Future Improvements

Add more features such as neighborhood ratings, crime rates, and nearby amenities.

Implement deep learning models for better predictions.

Author

Mohanesh Barge

For queries, reach out via GitHub or LinkedIn.


