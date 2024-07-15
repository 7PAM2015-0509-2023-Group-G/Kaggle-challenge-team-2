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
<!-- content -->Data was gathered from personal records from the damaged computer system of the Spaceship Titanic, which was damaged following a spacetime anomaly collision.​The dataset predicts if passengers were transferred to an alternate dimension.​
Passenger ID, HomePlanet, CryoSleep, cabin number, destination, age, VIP, costs, name, and whether or not they were transported to a different dimension are all included in the dataset. In addition, it displays name, age, VIP service, costs, and cabin numbers.​
Both numerical and categorical elements are included in the data​.There are 7 categorical values and 6 numerical values and 1 boolean value which is in the submission file.some records have missing values that need to be handled properly for modelling to be effective.​A passenger's ability to be moved to a different dimension is indicated by the binary target variable "Transported" (True/False)File and Data Field Descriptions
train.csv - Personal records for about two-thirds (~8700) of the passengers, to be used as training data.
PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
Destination - The planet the passenger will be debarking to.
Age - The age of the passenger.
VIP - Whether the passenger has paid for special VIP service during the voyage.
RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
Name - The first and last names of the passenger.
Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.
test.csv - Personal records for the remaining one-third (~4300) of the passengers, to be used as test data. Your task is to predict the value of Transported for the passengers in this set.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Exploratory Data Analysis
<!-- content -->
df.info and df.describe reveal that the dataset has null values and that the data types are not optimal.​

The distribution of the target variable, "Transported," is nearly balanced. This benefits training machine learning models from a balanced distribution, since it eliminates bias towards a certain class.​

The "age" feature nearly resembles a normal distribution, but all other numerical features show skewness. The box plots and df.describe corroborate this observation.

The "VIP" feature shows nearly equal distribution of each cases in which people were transported and cases in which they weren't.​This evenly distributed data indicates that the "VIP" feature has little effect on the target variable. We dropped the "VIP" feature from additional examination as a result.​

The "Num" feature's distribution plot shows that it has a substantial impact on the target variable.​

The "Deck" feature's value counts plot shows how the target variable is influenced differently by the many categories under this feature.​

The correlation heatmap shows that the numerical features do not significantly correlate with one another. Because there aren't many strong correlations between the numerical characteristics, it's likely that the features are independent of one another. ​

This makes machine learning models more effective by lowering multicollinearity and guaranteeing that each feature adds distinct information to the target variable's prediction.​

The 'group Count' vs 'Transported' count graphic makes it evident that Travellers in groups of two to seven are more likely to be transported.​

This implies that the quantity of passengers in a group affects the results of transportation. ​

Consequently, two new features, 'Group' and 'Groupcount', have been created to capture this relationship in the dataset.​

Based on the graph indicating lower transportation likelihood for group count 1, a new feature 'SoloTraveler' was created by checking if the 'Group' count was 1, capturing passengers traveling alone in the dataset.​
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
