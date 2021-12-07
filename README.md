### Hello! Welcome to PySpark Wine Prediction module!

#### Pull the latest docker image

`docker pull cemkaragozlu/spark-wine-prediction`

#### Run prediction on your data

`docker run -v /Users/keith/spark-wine-prediction-main/data/train.csv:/opt/spark_wine_prediction/test.csv wine-prediction python main.py --test-path test.csv`
Test data must be a semicolon separated csv file with necessary columns.
You only need to give the path to your test data in the indicated space.

docker run -v /Users/keith/spark-wine-prediction-main/data/train.csv:/opt/spark_wine_prediction/test.csv wine-prediction python main.py --test-path test.csv
