# Databricks notebook source
# MAGIC %md
# MAGIC # Value at risk - monte carlo
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
# MAGIC In this notebook, we use our 40 models created in previous stage and runs million of monte carlo simulation in parallel using **Apache Spark**. For each simulated market condition sampled from a multi variate distribution, we predict our hypothetical instrument returns. By storing all of our data back into **Delta Lake**, we create a data asset that can be queried on-demand (as opposition to end of day) across multiple down stream use cases

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP0` Control parameters

# COMMAND ----------

# DBTITLE 1,Import libraries
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import mlflow
import mlflow.sklearn
from datetime import datetime, timedelta
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *
from pyspark.sql import functions as F

# COMMAND ----------

# DBTITLE 1,Input parameters
portfolio_table = 'tania.ws_portfolio'
stock_table = 'tania.ws_stock'
stock_return_table = 'tania.ws_stock_return'
market_table = 'tania.ws_market'
market_return_table = 'tania.ws_market_return'
trial_table = 'tania.ws_monte_carlo'
trial_table_v2 = 'tania.ws_monte_carlo'

# when do we want to simulate data
trial_date = datetime.strptime('2020-05-11', '%Y-%m-%d')

# where did we log our model
model_path = '/tmp/models.json'

# how much history do we want compute volatility from
d_days = 90

# how many simulations do we want to run (industry standard ~ 20,000)
runs = 50000

# how many executors can run in parallel
parallelism = 12

# our predictive market factors
feature_names = ['SP500', 'NYSE', 'OIL', 'TREASURY', 'DOWJONES']

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP1` Retrieve models and data

# COMMAND ----------

pd.read_json('/tmp/models.json')

# COMMAND ----------

# DBTITLE 1,Load models
# models we serialized as json from pandas dataframe
# we load models as dictionary of instrument <-> weights
models = {}
for model in np.array(pd.read_json(model_path)):
    models[model[0]] = model[1]

model_dict = spark.sparkContext.broadcast(models)


# COMMAND ----------

# DBTITLE 1,Retrieve market factor data
def retrieve_market_factors(from_date, to_date):
    # Retrieve market factor returns in the provided time window
    from_ts = F.to_date(F.lit(from_date)).cast(TimestampType())
    to_ts = F.to_date(F.lit(to_date)).cast(TimestampType())
    f_ret = spark.table(market_return_table) \
        .filter(F.col('Date') > from_ts) \
        .filter(F.col('Date') <= to_ts) \
        .orderBy(F.asc('Date')) \
        .dropna()

    # Market factors easily fit in memory and will be used to create multivariate distribution of normal returns
    f_ret_pdf = f_ret.toPandas()
    f_ret_pdf.index = f_ret_pdf['Date']
    f_ret_pdf = f_ret_pdf.drop(['Date'], axis=1)
    return f_ret_pdf


# SAMPLE DATA
datapoint = datetime.fromisoformat("2020-03-01 10:10:10")
to_date = (datapoint).strftime("%Y-%m-%d")
from_date = (datapoint - timedelta(days=d_days)).strftime("%Y-%m-%d")
market_factors_df = retrieve_market_factors(from_date, to_date)
market_factors_df.head(10)


# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP2` Generate market conditions

# COMMAND ----------

# DBTITLE 1,Compute statistics
def compute_volatility(f_ret_pdf):
    # Retrieve market factor covariance matrix and average of returns
    # This will be used to generate a multi variate distribution of market factor returns
    return np.array(f_ret_pdf.mean()), np.array(f_ret_pdf.cov())


f_ret_avg, f_ret_cov = compute_volatility(market_factors_df)
f_ret_avg


# COMMAND ----------

# DBTITLE 1,Generate market conditions
# generate same feature vectors as used at model training phase
# add non linear transformations as simple example on non linear returns
def featurize(xs):
    fs = []
    for x in xs:
        fs.append(x)
        fs.append(np.sign(x) * x ** 2)
        fs.append(x ** 3)
        fs.append(np.sign(x) * np.sqrt(abs(x)))
    return fs


# provided covariance matrix and average of market factor, we sample from a multivariate distribution
# we allow a seed to be passed for reproducibility
def simulate_market(f_ret_avg_pdf, f_ret_cov_pdf, seed):
    np.random.seed(seed=seed)
    return np.random.multivariate_normal(f_ret_avg_pdf, f_ret_cov_pdf)


# provided simulated market factors and a specific model for an instrument,
# predict the instrument return in 2 weeks time
def predict(fs, ps):
    s = ps[0]
    for i, f in enumerate(fs):
        s = s + ps[i + 1] * f
    return float(s)


# COMMAND ----------

# DBTITLE 1,Example of reproducible market conditions
seed_init = 42
seeds = [seed_init + x for x in np.arange(0, 10)]
conditions = []
for seed in seeds:
    conditions.append(simulate_market(f_ret_avg, f_ret_cov, seed))

df = pd.DataFrame(conditions, columns=feature_names)
df['_seed'] = seeds
df


# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP3` Run monte-carlo

# COMMAND ----------

# DBTITLE 1,Run monte carlo simulations as pandas UDF
@pandas_udf('ticker string, seed int, trial float', PandasUDFType.GROUPED_MAP)
def run_trials(pdf):
    # Deserialize objects from cache
    models = model_dict.value
    f_ret_avg = f_ret_avg_B.value
    f_ret_cov = f_ret_cov_B.value

    trials = []
    for seed in np.array(pdf.seed):
        market_condition = simulate_market(f_ret_avg, f_ret_cov, seed)
        market_features = featurize(market_condition)
        for ticker in models.keys():
            trial = predict(market_features, models[ticker])
            trials.append([ticker, seed, trial])

    # Return a dataframe with each simulation across instruments per row
    trials_pdf = pd.DataFrame(data=trials, columns=['ticker', 'seed', 'trial'])
    return trials_pdf


# COMMAND ----------

# DBTITLE 1,Distribute trials
# Control experiment
to_date = trial_date.strftime("%Y-%m-%d")
from_date = (trial_date - timedelta(days=d_days)).strftime("%Y-%m-%d")
seed_init = int(trial_date.timestamp())

# create a dataframe of seeds so that each trial will result in a different simulation
# each executor is responsible for num_instruments * ( total_runs / num_executors ) trials
seed_pdf = pd.DataFrame([[seed_init + x, x % parallelism] for x in np.arange(0, runs)], columns=['seed', 'executor'])
seed_df = spark.createDataFrame(seed_pdf).repartition(parallelism, 'executor')
seed_df.cache()
seed_df.count()

# Compute volatility
market_df = retrieve_market_factors(from_date, to_date)
f_ret_avg, f_ret_cov = compute_volatility(market_df)
f_ret_avg_B = spark.sparkContext.broadcast(f_ret_avg)
f_ret_cov_B = spark.sparkContext.broadcast(f_ret_cov)

# group dataframe of seeds at the executor level and run simulations
mc_df = seed_df.groupBy('executor').apply(run_trials)

# store runs
mc_df = mc_df \
    .withColumn('run_date', F.lit(to_date)) \
    .join(spark.read.table(portfolio_table), 'ticker', 'inner') \
    .select('run_date', 'ticker', 'seed', 'trial', 'industry', 'country')

display(mc_df)

# COMMAND ----------

mc_df.write \
    .option('path', "dbfs:/mnt/mcdata/ws_monte_carlo_delta/") \
    .partitionBy("run_date") \
    .mode("append") \
    .format("delta") \
    .saveAsTable(trial_table)

# COMMAND ----------

# MAGIC %sql
# MAGIC select distinct(run_date) from tania.ws_monte_carlo

# COMMAND ----------

spark.read.table(trial_table).limit(100).toPandas()

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

