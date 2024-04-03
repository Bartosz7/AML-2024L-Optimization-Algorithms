# AML-2024L-Optimization-Algorithms
This project is one of the assignments for the 2024L Advanced Machine Learning course at the MiNI Faculty at Warsaw University of Technology (WUT).

### Datasets

* [S] [Diabetes](https://www.openml.org/search?type=data&sort=runs&status=active&qualities.NumberOfClasses=%3D_2&id=37)
* [S] [Tour & Travels Customer Churn](https://www.kaggle.com/datasets/tejashvi14/tour-travels-customer-churn-prediction)
* [S] [Seeds](https://archive.ics.uci.edu/dataset/236/seeds)
* [DISCARDED] ~~[S] [Employee](https://www.kaggle.com/datasets/tejashvi14/employee-future-prediction)~~
* [L] [League Of Legends Challenger Rank Game](https://www.kaggle.com/datasets/gyejr95/league-of-legends-challenger-rank-game10min15min)
* [L] [Jungle chess](https://www.openml.org/search?type=data&status=active&id=40997)
* [L] [Water quality](https://www.kaggle.com/datasets/mssmartypants/water-quality)
* [L] [Hotel Booking Cancellation](https://www.kaggle.com/datasets/youssefaboelwafa/hotel-booking-cancellation-prediction)
* [L] [Ionosphere](https://archive.ics.uci.edu/dataset/52/ionosphere)
* [L] [Sonar (Rock vs Mine)](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks)

### Setup
In order to use the code provided, please follow the following steps:

1. Clone the repository

2. Create (if not created yet) and activate a new Conda environment
```
conda create --name <new_env_name>
conda activate <new_env_name>
```

3. Import the exported environment file:
```
conda env import <path_to_exported_file>/environment.yml
```

4. Install any missing dependencies or packages:
```
conda install --file requirements.txt
```

5. Activate the newly create conda enviroment with installed dependencies. Use this environment when running jupyter notebooks and source code of the project.

