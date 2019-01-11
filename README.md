This project is a Kaggle competition: https://www.kaggle.com/c/avito-demand-prediction.
As beginners in machine learning,  we only used 10,000 records to build model. Each record has a picture.
We used ideas from Shivam Bansal(https://www.kaggle.com/shivamb/ideas-for-image-features-and-image-quality) and 
Ian Dzindo(https://github.com/Ian-Dzindo01/Avito-Demand-Prediction-Challenge) to extract image features and text features.
We used SVD to reduce dimension. We used Light GBM, Multilayer Perceptron (MLP), and Ridge Regression to predict demand.
We have three experiments: # Machine-Learning---avito
1. use all features to train models. 
2.1 use SVD to reduce dimension randomly. 
2.2 keep 26 features we thought were important, then use SVD on the rest features.
3. focus on Light GBM, reorder features according feature importance.
