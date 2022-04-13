import time

from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

import math
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.functions import udf, pandas_udf, PandasUDFType


class MonteCarlo():
    def __init__(self, conf: Dict[str, Any], spark: SparkSession, experiment_id: str):
        self.conf = conf
        self.spark = spark
        self.experiment_id = experiment_id

    def split_data_to_date(self, market_return_table_path: str, stock_return_table_path: str):
        split_date = F.to_date(F.lit(self.conf['date_str'])).cast(TimestampType())

        # retrieve historical tick data up to specified date
        f_ret = self.spark.read.table(market_return_table_path).filter(F.col('Date') <= split_date)
        s_ret = self.spark.read.table(stock_return_table_path).filter(F.col('Date') <= split_date)
        mlflow.log_param('to_date', self.conf['date_str'])

        return f_ret, s_ret

    def calc_correlation(self, f_ret):
        # market factors easily fit in memory and are required to build normal distribution of returns
        f_ret_pdf = f_ret.toPandas()
        f_ret_pdf.index = f_ret_pdf['Date']
        f_ret_pdf = f_ret_pdf.drop(['Date'], axis=1)

        f_cor_pdf = f_ret_pdf.corr(method='spearman', min_periods=12)
        sns.set(rc={'figure.figsize': (11, 8)})
        sns.heatmap(f_cor_pdf, annot=True)
        plt.savefig('/tmp/factor_correlation.png')
        mlflow.log_metric('x_size', f_ret_pdf.size)
        return

    def create_train(self, correlation=True):
        # creating training dataset based on market factors
        # create our feature set based on market factors and actual portfolio return
        # in real life, we should obviously split set into training / testing
        f_ret, s_ret = self.split_data_to_date(self.conf['market_return_table'], self.conf['stock_return_table'])
        if correlation:
            self.calc_correlation(f_ret)
        feature_names = self.conf['feature_names'].split(" ")
        x_train = f_ret \
            .withColumn("features", F.array(feature_names)) \
            .dropna() \
            .select('date', 'features') \
            .join(s_ret, 'date')
        return x_train

    def train(self):
        x_train = self.create_train()
        # use pandas UDF to train multiple model (one for each instrument) in parallel
        # the resulting dataframe will be the linear regression weights for each instrument
        schema = StructType(
            [StructField('ticker', StringType(), True), StructField('weights', ArrayType(FloatType()), True)])

        def featurize(xs):
            # add non linear transformations as simple example on non linear returns
            fs = []
            for x in xs:
                fs.append(x)
                fs.append(np.sign(x) * x ** 2)
                fs.append(x ** 3)
                fs.append(np.sign(x) * np.sqrt(abs(x)))
            return fs

        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def train_model(group, pdf):
            # build market factor vectors
            # add a constant - the intercept term for each instrument i.
            X = [featurize(row) for row in np.array(pdf['features'])]
            X = sm.add_constant(X, prepend=True)
            y = np.array(pdf['return'])
            model = sm.OLS(y, X).fit()
            w_df = pd.DataFrame(data=[[model.params]], columns=['weights'])
            w_df['ticker'] = group[0]
            return w_df

        # the resulting dataframe easily fits in memory and will be saved as our "uber model", serialized to json
        models_df = x_train.groupBy('ticker').apply(train_model).toPandas()
        models_df.to_json('/tmp/models.json')
        return

    def predict(self):
        x_train = self.create_train()
        # simply applying weight to each market factor feature

        def featurize(xs):
            # add non linear transformations as simple example on non linear returns
            fs = []
            for x in xs:
                fs.append(x)
                fs.append(np.sign(x) * x ** 2)
                fs.append(x ** 3)
                fs.append(np.sign(x) * np.sqrt(abs(x)))
            return fs

        @udf("float")
        def predict_udf(xs, ps):
            fs = featurize(xs)
            s = ps[0]
            for i, f in enumerate(fs):
                s = s + ps[i + 1] * f
            return float(s)

        # Compute mean square error
        @udf("float")
        def wsse_udf(p, a):
            return float((p - a) ** 2)

        # we read models created at previous step
        models_df = self.spark.createDataFrame(pd.read_json("/tmp/models.json"))

        # we join model for each return to compute prediction of return vs. actual
        prediction_df = x_train.join(models_df, ['ticker']) \
            .withColumn("predicted", predict_udf(F.col('features'), F.col('weights'))) \
            .withColumnRenamed('return', 'actual') \
            .select('ticker', 'Date', 'predicted', 'actual')

        # compare expected vs. actual return
        # sum mean square error per instrument
        wsse_df = prediction_df \
            .withColumn('wsse', wsse_udf(F.col('predicted'), F.col('actual'))) \
            .groupBy('ticker') \
            .agg(F.sum('wsse')) \
            .toPandas()

        # plot mean square error as accuracy of our model for each instrument
        ax = wsse_df.plot.bar(x='ticker', y='sum(wsse)', rot=0, label=None, figsize=(24, 5))
        ax.get_legend().remove()
        plt.title("Model WSSE for each instrument")
        plt.xticks(rotation=45)
        plt.ylabel("wsse")
        plt.savefig("/tmp/model_wsse.png")

        # DBTITLE 1,Show predictive value for [Ecopetrol S.A.], Oil & Gas Producers in Columbia
        df = prediction_df.filter(F.col('ticker') == "EC").toPandas()
        plt.figure(figsize=(20, 8))
        plt.plot(df.Date, df.actual)
        plt.plot(df.Date, df.predicted, color='green', linestyle='--')
        plt.title('Log return of EC')
        plt.ylabel('log return')
        plt.xlabel('Date')
        plt.savefig("/tmp/model_prediction.png")


        mlflow.log_artifact('/tmp/model_wsse.png')
        mlflow.log_artifact('/tmp/factor_correlation.png')
        mlflow.log_artifact('/tmp/model_prediction.png')
        mlflow.log_artifact('/tmp/models.json')

        return

