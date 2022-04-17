# Databricks notebook source
# MAGIC %md
# MAGIC # Value at risk - aggregation
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
# MAGIC In this notebook, we demonstrate the versatile nature of our model carlo simulation on **Delta Lake**. Stored in its most granular form, analysts have the flexibility to slice and dice their data to aggregate value-at-risk on demand via a user aggregated defined function on **Spark SQL**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP0` Configuration

# COMMAND ----------

# DBTITLE 1,Import libraries
# MAGIC %matplotlib inline
# MAGIC import pandas as pd
# MAGIC import numpy as np
# MAGIC import matplotlib.pyplot as plt
# MAGIC from scipy import stats
# MAGIC import time
# MAGIC from datetime import datetime, timedelta
# MAGIC from pyspark.sql.functions import pandas_udf, PandasUDFType
# MAGIC from pyspark.sql.window import Window
# MAGIC from pyspark.sql.types import *
# MAGIC from pyspark.sql import functions as F

# COMMAND ----------

# DBTITLE 1,Control parameters
portfolio_table = 'tania.ws_portfolio'
stock_table = 'tania.ws_stock'
stock_return_table = 'tania.ws_stock_return'
market_table = 'tania.ws_market'
market_return_table = 'tania.ws_market_return'
trial_table = 'tania.ws_monte_carlo'

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from tania.ws_monte_carlo

# COMMAND ----------

# DBTITLE 1,Control parameters
# MAGIC %scala
# MAGIC 
# MAGIC //number of simulations
# MAGIC val runs = 50000
# MAGIC 
# MAGIC //value at risk confidence
# MAGIC val confidenceVar = 95

# COMMAND ----------

# DBTITLE 1,Useful widgets
try:
  dbutils.widgets.remove('run')
except:
  print('No widget named [run]')

all_runs = sql("SELECT DISTINCT run_date FROM {}".format(trial_table)).toPandas()['run_date']
dbutils.widgets.dropdown("run", all_runs[0], all_runs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP1` Compute value at risk

# COMMAND ----------

# DBTITLE 1,Compute point in time value at risk
run = dbutils.widgets.get('run')

# aggregate monte carlo simulations
mc_df = spark \
  .read \
  .table(trial_table) \
  .filter(F.col('run_date') == run) \
  .withColumnRenamed('trial', 'return') \
  .groupBy('seed') \
  .agg(F.sum('return').alias('return')) \
  .select('return') \
  .toPandas()

returns = mc_df['return']

# compute 95 value at risk
value_at_risk = np.quantile(returns, 5 / 100)
mc_df['exceeds'] = mc_df['return'] > value_at_risk

# extract normal distribution
avg = returns.mean()
std = returns.std()
x1 = np.arange(returns.min(),returns.max(),0.01)
y1 = stats.norm.pdf(x1, loc=avg, scale=std)
x2 = np.arange(returns.min(),value_at_risk,0.001)
y2 = stats.norm.pdf(x2, loc=avg, scale=std)

# plot value at risk
ax = mc_df.hist(column='return', bins=50, density=True, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
ax = ax[0]
for x in ax:
  
    # Despine
    x.spines['right'].set_visible(False)
    x.spines['top'].set_visible(False)
    x.spines['left'].set_visible(False)

    x.axvline(x=value_at_risk, color='r', linestyle='dashed', linewidth=1)
    x.fill_between(x2, y2, zorder=3, alpha=0.4)
    x.plot(x1, y1, zorder=3)
    
    # Switch off ticks
    x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

    # Draw horizontal axis lines
    vals = x.get_yticks()
    for tick in vals:
        x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    x.text(value_at_risk - 0.2, 1, "VAR(95) = {:2f}".format(value_at_risk), rotation=90)
    x.set_title('')
    x.set_ylabel('')
    #x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))


# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP2` Slice and dice value at risk

# COMMAND ----------

# DBTITLE 1,Create user aggregated function for "on demand VAR calculation"
# MAGIC %scala
# MAGIC 
# MAGIC import org.apache.spark.sql.SparkSession
# MAGIC import org.apache.spark.sql.expressions.MutableAggregationBuffer
# MAGIC import org.apache.spark.sql.expressions.UserDefinedAggregateFunction
# MAGIC import org.apache.spark.sql.Row
# MAGIC import org.apache.spark.sql.types._
# MAGIC 
# MAGIC class ValueAtRisk(n: Int) extends UserDefinedAggregateFunction {
# MAGIC 
# MAGIC   // This is the input fields for your aggregate function.
# MAGIC   override def inputSchema: org.apache.spark.sql.types.StructType = StructType(StructField("value", DoubleType) :: Nil)
# MAGIC 
# MAGIC   // This is the internal fields you keep for computing your aggregate.
# MAGIC   override def bufferSchema: StructType = StructType(Array(StructField("worst", ArrayType(DoubleType))))
# MAGIC 
# MAGIC   // This is the output type of your aggregatation function.
# MAGIC   override def dataType: DataType = DoubleType
# MAGIC 
# MAGIC   // The order we process dataframe does not matter, the worst will always be the worst
# MAGIC   override def deterministic: Boolean = true
# MAGIC 
# MAGIC   // This is the initial value for your buffer schema.
# MAGIC   override def initialize(buffer: MutableAggregationBuffer): Unit = {
# MAGIC     buffer(0) = Seq.empty[Double]
# MAGIC   }
# MAGIC 
# MAGIC   // This is how to update your buffer schema given an input.
# MAGIC   override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
# MAGIC     buffer(0) = buffer.getAs[Seq[Double]](0) :+ input.getAs[Double](0)
# MAGIC   }
# MAGIC 
# MAGIC   // This is how to merge two objects with the bufferSchema type.
# MAGIC   // We only keep worst N events
# MAGIC   override def merge(buffer: MutableAggregationBuffer, row: Row): Unit = {
# MAGIC     buffer(0) = (buffer.getAs[Seq[Double]](0) ++ row.getAs[Seq[Double]](0)).sorted.take(n)
# MAGIC   }
# MAGIC 
# MAGIC   // This is where you output the final value, given the final value of your bufferSchema.
# MAGIC   // Our value at risk is best of the worst n overall
# MAGIC   override def evaluate(buffer: Row): Any = {
# MAGIC     return buffer.getAs[Seq[Double]](0).sorted.last
# MAGIC   }
# MAGIC 
# MAGIC }
# MAGIC 
# MAGIC // Assume we've generated 50,000 monte-carlo simulations for each instrument
# MAGIC val numRecords = runs
# MAGIC 
# MAGIC // We want to compute Var(95)
# MAGIC val confidence = confidenceVar
# MAGIC 
# MAGIC // So the value at risk is the best of the worst N events 
# MAGIC val n = (100 - confidence) * numRecords / 100
# MAGIC 
# MAGIC // Register UADFs
# MAGIC val valueAtRisk = new ValueAtRisk(n)
# MAGIC spark.udf.register("VALUE_AT_RISK", new ValueAtRisk(n))

# COMMAND ----------

# DBTITLE 1,Total number of simulations
# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM tania.ws_monte_carlo

# COMMAND ----------

# DBTITLE 1,Show evolution of value at risk
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE tania.ws_total_var AS 
# MAGIC SELECT 
# MAGIC   t.run_date AS day, 
# MAGIC   VALUE_AT_RISK(t.return) AS value_at_risk
# MAGIC FROM 
# MAGIC   (
# MAGIC   SELECT 
# MAGIC     m.run_date, 
# MAGIC     sum(m.trial) AS return
# MAGIC   FROM
# MAGIC     tania.ws_monte_carlo m
# MAGIC   GROUP BY
# MAGIC     m.run_date, 
# MAGIC     m.seed 
# MAGIC   ) t
# MAGIC GROUP BY 
# MAGIC   t.run_date
# MAGIC ORDER BY t.run_date ASC;
# MAGIC 
# MAGIC SELECT * FROM tania.ws_total_var where day<"2020-05-03"

# COMMAND ----------

# DBTITLE 1,Risk exposure to different countries
# MAGIC %sql
# MAGIC 
# MAGIC CREATE OR REPLACE TABLE tania.ws_country_var AS 
# MAGIC SELECT 
# MAGIC   t.run_date AS day, 
# MAGIC   LOWER(t.country) AS country,
# MAGIC   VALUE_AT_RISK(t.return) AS value_at_risk
# MAGIC FROM 
# MAGIC   (
# MAGIC   SELECT 
# MAGIC     m.run_date, 
# MAGIC     m.seed, 
# MAGIC     m.country, 
# MAGIC     sum(m.trial) AS return
# MAGIC   FROM
# MAGIC     tania.ws_monte_carlo m
# MAGIC   GROUP BY
# MAGIC     m.run_date, 
# MAGIC     m.seed, 
# MAGIC     m.country
# MAGIC   ) t
# MAGIC GROUP BY 
# MAGIC   t.run_date, 
# MAGIC   t.country
# MAGIC ORDER BY t.run_date ASC;
# MAGIC 
# MAGIC SELECT * FROM tania.ws_country_var where day<"2020-05-03";

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from tania.ws_peru_var

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from tania.ws_peru_var_contribution

# COMMAND ----------

# DBTITLE 1,Peru seems to have biggest risk exposure, aggregate by industry
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE tania.ws_peru_var AS 
# MAGIC SELECT 
# MAGIC   t.run_date AS day, 
# MAGIC   LOWER(t.industry) AS industry,
# MAGIC   VALUE_AT_RISK(t.return) AS value_at_risk
# MAGIC FROM 
# MAGIC   (
# MAGIC   SELECT 
# MAGIC     m.run_date, 
# MAGIC     m.seed, 
# MAGIC     m.industry, 
# MAGIC     sum(m.trial) AS return
# MAGIC   FROM
# MAGIC     tania.ws_monte_carlo m
# MAGIC   WHERE
# MAGIC     m.country = 'PERU'
# MAGIC   GROUP BY
# MAGIC     m.run_date, 
# MAGIC     m.seed, 
# MAGIC     m.industry
# MAGIC   ) t
# MAGIC GROUP BY 
# MAGIC   t.run_date, 
# MAGIC   t.industry
# MAGIC ORDER BY t.run_date ASC;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE tania.ws_peru_var_contribution AS
# MAGIC SELECT c.day, c.industry, c.value_at_risk / t.value_at_risk AS contribution
# MAGIC FROM tania.ws_peru_var c
# MAGIC JOIN (
# MAGIC   SELECT day, SUM(value_at_risk) AS value_at_risk
# MAGIC   FROM tania.ws_peru_var
# MAGIC   GROUP BY day
# MAGIC ) t
# MAGIC WHERE c.day = t.day;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   c.day,
# MAGIC   c.industry,
# MAGIC   ABS(c.contribution * t.value_at_risk) AS contribution
# MAGIC FROM tania.ws_peru_var_contribution c
# MAGIC JOIN tania.ws_total_var t
# MAGIC WHERE t.day = c.day 
# MAGIC AND t.day<"2020-05-03"
# MAGIC ORDER BY t.day;

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


