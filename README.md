### Hello! Welcome to PySpark Wine Prediction module!

#### Pull the latest docker image

`docker pull sshower111/wine-prediction`

#### Run prediction on your data

`docker run -v <path-to-test-data>:/opt/wine_prediction/test.csv wine-prediction python main.py --test-path test.csv`
Test data must be a semicolon separated csv file with necessary columns.
You only need to give the path to your test data in the indicated space.
