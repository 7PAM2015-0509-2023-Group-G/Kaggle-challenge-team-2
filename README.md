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

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## X-AI
<!-- content -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Conclusion
<!-- content -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>



We hope you find our project insightful and useful!
