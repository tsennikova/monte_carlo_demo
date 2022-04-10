# Databricks notebook source
# MAGIC %md
# MAGIC # Value at risk - modeling
# MAGIC
# MAGIC **Modernizing risk management practice**: *Traditional banks relying on on-premises infrastructure can no longer effectively manage risk. Banks must abandon the computational inefficiencies of legacy technologies and build an agile Modern Risk Management practice capable of rapidly responding to market and economic volatility. Using value-at-risk use case, you will learn how Databricks is helping FSIs modernize their risk management practices, leverage Delta Lake, Apache Spark and MLFlow to adopt a more agile approach to risk management.*
# MAGIC
# MAGIC ---
# MAGIC + <a href="$./00_var_context">STAGE0</a>: Home page
# MAGIC + <a href="$./01_var_market_etl">STAGE1</a>: Using Delta Lake for a curated and a 360 view of your risk portfolio
# MAGIC + <a href="$./02_var_model">STAGE2</a>: Tracking experiments and registering risk models through MLflow capabilities
# MAGIC + <a href="$./03_var_monte_carlo">STAGE3</a>: Leveraging the power of Apache Spark for massively distributed Monte Carlo simulations
# MAGIC + <a href="$./04_var_aggregation">STAGE4</a>: Slicing and dicing through your risk exposure using collaborative notebooks and SQL
# MAGIC + <a href="$./05_var_alt_data">STAGE5</a>: Acquiring news analytics data as a proxy of market volatility
# MAGIC + <a href="$./06_var_backtesting">STAGE6</a>: Reporting breaches through model risk backtesting
# MAGIC ---
# MAGIC <antoine.amend@databricks.com>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Context
# MAGIC In this notebook, we retrieve last 2 years worth of market indicator data to train a model that could predict our instrument returns. As our portfolio is made of 40 equities, we want to train 40 predictive models in parallel, collecting all weights into a single coefficient matrix for monte carlo simulations. We show how to have a more discipline approach to model development by leveraging **MLFlow** capabilities.

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP0` Configuration

# COMMAND ----------

# DBTITLE 1,Import libraries
%matplotlib inline

import pandas as pd
import math
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import mlflow

from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
from datetime import datetime, timedelta
# COMMAND ----------

# DBTITLE 1,Control parameters
portfolio_table = 'tania.ws_portfolio'
stock_table = 'tania.ws_stock'
stock_return_table = 'tania.ws_stock_return'
market_table = 'tania.ws_market'
market_return_table = 'tania.ws_market_return'

# when do we train model
today_str = "2019-06-01"
today = F.to_date(F.lit(today_str)).cast(TimestampType())

# our predictive market factors
feature_names = ['SP500', 'NYSE', 'OIL', 'TREASURY', 'DOWJONES']

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP1` Access data

# COMMAND ----------

# DBTITLE 1,Delta lake ensures full reproducibility
versions_m_df = sql("DESCRIBE HISTORY " + market_return_table).select("version")
delta_m_version = versions_m_df.toPandas()['version'].max()

versions_s_df = sql("DESCRIBE HISTORY " + stock_return_table).select("version")
delta_s_version = versions_s_df.toPandas()['version'].max()

# COMMAND ----------

# DBTITLE 1,Get stocks and market factor returns
# retrieve historical tick data up to specified date
f_ret = spark.table(market_return_table).filter(F.col('Date') <= today)
s_ret = spark.table(stock_return_table).filter(F.col('Date') <= today)

# market factors easily fit in memory and are required to build normal distribution of returns
f_ret_pdf = f_ret.toPandas()
f_ret_pdf.index = f_ret_pdf['Date']
f_ret_pdf = f_ret_pdf.drop(['Date'], axis=1)
f_ret_pdf.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ##`STEP2` Evaluate market factors

# COMMAND ----------

# DBTITLE 1,Correlation of market factors
# we simply plot correlation matrix via pandas (market factors fit in memory)
# we assume market factors are not correlated (NASDAQ and SP500 are, so are OIL and TREASURY BONDS)
f_cor_pdf = f_ret_pdf.corr(method='spearman', min_periods=12)
sns.set(rc={'figure.figsize': (11, 8)})
sns.heatmap(f_cor_pdf, annot=True)
plt.savefig('/tmp/factor_correlation.png')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##`STEP3` Train a model for each instrument

# COMMAND ----------

# DBTITLE 1,Creating training dataset based on market factors
# create our feature set based on market factors and actual portfolio return
# in real life, we should obviously split set into training / testing
x_train = f_ret \
    .withColumn("features", F.array(feature_names)) \
    .dropna() \
    .select('date', 'features') \
    .join(s_ret, 'date')

display(x_train)


# COMMAND ----------

# DBTITLE 1,Train models in parallel using pandas UDF
# add non linear transformations as simple example on non linear returns
def featurize(xs):
    fs = []
    for x in xs:
        fs.append(x)
        fs.append(np.sign(x) * x ** 2)
        fs.append(x ** 3)
        fs.append(np.sign(x) * np.sqrt(abs(x)))
    return fs


# use pandas UDF to train multiple model (one for each instrument) in parallel
# the resulting dataframe will be the linear regression weights for each instrument
schema = StructType([StructField('ticker', StringType(), True), StructField('weights', ArrayType(FloatType()), True)])


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
models_df.to_json("/tmp/models.json")


# COMMAND ----------

# DBTITLE 1,Predict daily returns
# simply applying weight to each market factor feature
@udf("float")
def predict_udf(xs, ps):
    fs = featurize(xs)
    s = ps[0]
    for i, f in enumerate(fs):
        s = s + ps[i + 1] * f
    return float(s)


# we read models created at previous step
models_df = spark.createDataFrame(pd.read_json("/tmp/models.json"))

# we join model for each return to compute prediction of return vs. actual
prediction_df = x_train.join(models_df, ['ticker']) \
    .withColumn("predicted", predict_udf(F.col('features'), F.col('weights'))) \
    .withColumnRenamed('return', 'actual') \
    .select('ticker', 'Date', 'predicted', 'actual')


# COMMAND ----------

# DBTITLE 1,Compute mean square error
@udf("float")
def wsse_udf(p, a):
    return float((p - a) ** 2)


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
plt.show()

# COMMAND ----------

# DBTITLE 1,Show predictive value for [Ecopetrol S.A.], Oil & Gas Producers in Columbia
df = prediction_df.filter(F.col('ticker') == "EC").toPandas()
plt.figure(figsize=(20, 8))
plt.plot(df.Date, df.actual)
plt.plot(df.Date, df.predicted, color='green', linestyle='--')
plt.title('Log return of EC')
plt.ylabel('log return')
plt.xlabel('Date')
plt.savefig("/tmp/model_prediction.png")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP4` register model

# COMMAND ----------

# DBTITLE 1,Log model and artifacts to MLflow
mlflow.set_experiment("/Users/tatiana.sennikova@databricks.com/Monte Carlo Experiment")
with mlflow.start_run(run_name='MC_RUN') as parent_run:
    mlflow.log_param('to_date', today_str)
    mlflow.log_metric('x_size', f_ret_pdf.size)
    mlflow.log_param('delta.version.market', delta_m_version)
    mlflow.log_param('delta.version.stocks', delta_s_version)
    mlflow.log_artifact('/tmp/model_wsse.png')
    mlflow.log_artifact('/tmp/factor_correlation.png')
    mlflow.log_artifact('/tmp/model_prediction.png')
    mlflow.log_artifact('/tmp/models.json')
    mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC + <a href="$./00_var_context">STAGE0</a>: Home page
# MAGIC + <a href="$./01_var_market_etl">STAGE1</a>: Using Delta Lake for a curated and a 360 view of your risk portfolio
# MAGIC + <a href="$./02_var_model">STAGE2</a>: Tracking experiments and registering risk models through MLflow capabilities
# MAGIC + <a href="$./03_var_monte_carlo">STAGE3</a>: Leveraging the power of Apache Spark for massively distributed Monte Carlo simulations
# MAGIC + <a href="$./04_var_aggregation">STAGE4</a>: Slicing and dicing through your risk exposure using collaborative notebooks and SQL
# MAGIC + <a href="$./05_var_alt_data">STAGE5</a>: Acquiring news analytics data as a proxy of market volatility
# MAGIC + <a href="$./06_var_backtesting">STAGE6</a>: Reporting breaches through model risk backtesting
# MAGIC ---

# COMMAND ----------

