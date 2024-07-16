<a id="readme-top"></a>

<!-- TEAM LOGO -->
<br />
<div align="left">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="TEAM 2.png" alt="Logo" width="200" height="200">
  </a>
</div>

# Team 2 - Kaggle Challenge

This repository contains the work of Team 2 for Assignment 2 - Kaggle Challenge, part of our coursework at the [University of Hertfordshire, UK](https://www.herts.ac.uk/).

Team Members:

- **Bineeth Mathew** - [GitHub](https://github.com/Bineethmathew)
- **Anns Tomy** - [GitHub](https://github.com/AnnsTomy)
- **Dhanya Davis** - [GitHub](https://github.com/dhanyadavis1999)
- **Meenakshi Rajesh** - [GitHub](https://github.com/Meenakshi-Rajesh)
- **Gobu Babu** - [GitHub](https://github.com/gobucbabu)
- **Aaron Joseph** - [GitHub](https://github.com/aaronmj7)

<!-- [Contributions](https://github.com/7PAM2015-0509-2023-Group-G/Kaggle-challenge-team-2/graphs/contributors) -->



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#project-overview">Project Overview</a>
    </li>
    <li>
      <a href="#objectives">Objectives</a>
    </li>
    <li>
      <a href="#data">Data</a>
    </li>
    <li>
      <a href="#exploratory-data-analysis">Exploratory Data Analysis</a>
    </li>
    <li>
      <a href="#pre-processing">Pre-Processing</a>
    </li>
    <li>
      <a href="#models">Models</a>
    </li>
    <li>
      <a href="#x-ai">X-AI</a>
    </li>
    <li>
      <a href="#conclusion">Conclusion</a>
    </li>
  </ol>
</details>



## Project Overview

This is a group assignment where students will participate in a Kaggle competition: [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic). Our task is to develop a statistical/machine learning model to predict which passengers of the Spaceship Titanic were transported by a spacetime anomaly, using records recovered from the spaceship’s damaged computer system. We will submit our model results to Kaggle to obtain a score and present our findings in a group presentation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Objectives

- **Data Exploration and Visualization**: Explore and visualize the dataset.
- **Data Pre-processing**: Build a pipeline to pre-process the dataset.
- **Model Training**: Train and customize a viable model.
- **Collaboration**: Use GitHub and Google Colab for a collaborative research environment.
- **Presentation**: Present our findings in a group presentation with a PowerPoint presentation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Built With
* ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
* ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
* ![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
* ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
* ![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
* ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
* ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)



## Data
<!-- content -->
Data was collected from personal records stored in the damaged computer system of the Spaceship Titanic, which suffered damage following a collision with a spacetime anomaly. The dataset aims to predict whether passengers were transferred to an alternate dimension.

The dataset includes variables such as Passenger ID, HomePlanet, CryoSleep status, cabin number, destination, age, VIP status, costs, name, and whether the passenger was transported to another dimension. It contains both numerical and categorical elements, with 7 categorical values, 6 numerical values, and 1 boolean value in the submission file. Some records have missing values that need to be addressed for effective modeling. The binary target variable "Transported" (True/False) indicates whether a passenger was moved to a different dimension.

**File and Data Field Descriptions:**
- **train.csv**: Contains personal records for approximately two-thirds (~8700) of the passengers, used as training data.
  - **PassengerId**: A unique identifier for each passenger, formatted as gggg_pp, where gggg indicates a group and pp is the passenger's number within the group. Group members are often family members but not always.
  - **HomePlanet**: The planet from which the passenger departed, typically their permanent residence.
  - **CryoSleep**: Indicates if the passenger chose to be in suspended animation for the voyage, confining them to their cabins.
  - **Cabin**: The cabin number where the passenger stayed, formatted as deck/num/side, with side being either P (Port) or S (Starboard).
  - **Destination**: The planet where the passenger will disembark.
  - **Age**: The age of the passenger.
  - **VIP**: Indicates if the passenger paid for VIP service during the voyage.
  - **RoomService, FoodCourt, ShoppingMall, Spa, VRDeck**: Amounts billed by the passenger at various luxury amenities on the Spaceship Titanic.
  - **Name**: The first and last names of the passenger.
  - **Transported**: The target variable indicating whether the passenger was transported to another dimension.

- **test.csv**: Contains personal records for the remaining one-third (~4300) of the passengers, used as test data. The task is to predict the "Transported" value for these passengers.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Exploratory Data Analysis
<!-- content -->
The dataset analysis revealed several key insights. Firstly, the dataset contains null values and suboptimal data types, as indicated by `df.info` and `df.describe`. This necessitates proper handling of missing values and optimization of data types for effective modeling.

The target variable "Transported" is nearly balanced, which is advantageous for training machine learning models as it reduces bias towards any particular class. This balanced distribution ensures that the model can learn to predict both classes effectively.

Regarding feature distributions, the "age" feature approximates a normal distribution, while other numerical features exhibit skewness, confirmed by box plots and `df.describe`. The "VIP" feature shows an almost equal distribution between transported and non-transported cases, suggesting it has minimal impact on the target variable. Consequently, the "VIP" feature was excluded from further analysis. The "Num" feature significantly influences the target variable, as shown by its distribution plot. Additionally, the "Deck" feature's value counts plot indicates varying impacts on the target variable across its categories.

The correlation heatmap shows that numerical features do not significantly correlate with each other, suggesting feature independence. This reduces multicollinearity and ensures each feature contributes unique information to the target variable's prediction. The 'group Count' vs 'Transported' count graphic indicates that passengers in groups of two to seven are more likely to be transported. This led to creation of two new features, 'Group' and 'Groupcount', to capture this relationship.

Based on the observation that solo travelers (group count of 1) are less likely to be transported, a new feature 'SoloTraveler' was created to identify passengers traveling alone. These insights and feature engineering steps enhance the dataset's suitability for effective machine-learning modeling.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Pre-Processing
<!-- content -->
Preprocessing involves converting raw data into a clean format, which is essential for enhancing the performance and accuracy of machine learning models. Five preprocessing methods were used in this analysis: Dropping Features, Encoding, Scaling, Imputation, and Principal Component Analysis (PCA). 

The Dropping Features method was employed to remove irrelevant features, resulting in the exclusion of two columns, 'Passenger ID' and 'Name', since Passenger IDs are unique for each record and 'Name' had only 220 non-unique values.
One-hot encoding was used to convert Boolean and categorical features into integers to simplify the analysis. The columns 'CryoSleep' and 'Transported' contained Boolean values, while 'HomePlanet', 'Destination', 'Deck', and 'Side' were categorical, and all were converted to integers to reduce complexity.
StandardScaler was applied to normalize the data, enhancing model efficiency. This standardization process ensures that each feature has a mean of zero and a standard deviation of one, allowing them to contribute equally to the model’s performance.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Models
<!-- content -->

Among the six popular classification models tested by each group member—Random Forest, Support Vector Classification (SVC), Decision Tree, Logistic Regression, XGBoost, and K-Nearest Neighbors (KNN)—SVC achieved the highest accuracy. SVC is a supervised machine learning algorithm specifically designed for classification tasks. Its primary objective is to maximize the margin, which is the distance between the hyperplane (the decision boundary) and the nearest data points, known as support vectors, in an N-dimensional feature space. This maximization helps in achieving better separation between different classes.

To optimize the performance of the SVC model, you conducted model tuning. This involved determining the ideal hyperparameters, such as the regularization parameter (C), the kernel type (e.g., linear, polynomial, radial basis function), and the kernel coefficient (gamma). Cross-validation was employed to systematically evaluate different combinations of these hyperparameters and select the best ones.

Additionally, you applied various preprocessing methods and hyperparameter tuning techniques for each Kaggle submission. This iterative process helped in refining the model and improving its performance on the given datasets.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



## X-AI
<!-- content -->
CryoSleep is identified as the most important feature, having nearly twice the weight of the next most important feature, VRDeck. This indicates that CryoSleep has a significant impact on the model’s predictions. Food Court and Spa have similar importance values, suggesting they contribute equally to the model’s decisions. The Age feature also has a reasonable level of importance, indicating it plays a notable role in classification.
On the other hand, the remaining features have relatively small importance values, which are steadily declining. This means they contribute less to the model’s predictions. Additionally, some features appear to have zero importance, indicating they do not influence the model’s decisions at all.
This analysis helps in understanding which features are most influential in your SVC model and can guide further feature selection or engineering efforts to improve model performance.


Logistic Regression model, Spa and VRDeck are the most important features, followed by RoomService and FoodCourt. Other features, except for ShoppingMall, have smaller importance. The CryoSleep feature shows a clear separation between higher and lower points, indicating its distinct impact on the model’s predictions.


A comparison of the Shapley summary plots for XGBoost, Decision Tree, and Random Forest models, all of which are based on decision trees, shows that Spa, VRDeck, RoomService, and FoodCourt are the predominant features. However, there are no recognizable patterns in the behavior of the remaining features, though most features do have some relative importance.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

##  Result
<!-- content -->
XAI analysis reveals that the features CryoSleep, Spa, VRDeck, RoomService, FoodCourt, and ShoppingMall have higher importance across all models, with varying significance. EDA showed that five of these six numerical features (except CryoSleep) have skewed distributions, likely contributing to their importance. The CryoSleep feature, being binary, shows a clear gap between high and low importance points.

The highest Kaggle submission score was 0.80149, ranking at position 544. There were 65 submissions in total, with all models showing significant improvement from their initial scores. The SVC model had the highest score, improving from 0.7814 to 0.8001 after hyperparameter tuning and additional preprocessing. This model appears to accurately predict when passengers are transported, as it has fewer false negatives, 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Conclusion
<!-- content -->
- All chosen models improved their initial submission scores upon hyperparameter tuning and
additional pre-processing. All chosen models performed our prediction tasks in a considerably
well manner.
- Tree based approaches showed some similar behaviour, specifically in the feature importance for
those models, while the rest of the approaches were fairly distinct.
- Six out of the fourteen features seem to have a higher prediction importance in all models, as
revealed by the XAI findings. Five of these are the amounts spent on luxury amenities and the
remaining one being cryosleep.
- SVC outperformed all other models with a higher score on Kaggle, which was improved upon
from its initial score of 78.14% to 80.15%. It accurately predicts transported passengers with
fewer False Negatives, suggesting it is well-suited for this task.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



We hope you find our project insightful and useful!
