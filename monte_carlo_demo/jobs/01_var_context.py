# Databricks notebook source
# MAGIC %md
# MAGIC # Value at risk
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

import numpy as np

# COMMAND ----------

# COMMAND ----------

displayHTML("""<iframe src="https://www.youtube.com/embed/d8t0Y8leE1c?&mute=1"></iframe>""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Context
# MAGIC In this demo, we demonstrate how banks can modernize their risk management practices by efficiently scaling their Monte Carlo simulations from tens of thousands up to millions by leveraging both the flexibility of cloud compute and the robustness of Apache Spark. We show how Databricks, as the only Unified Data Analytics Platform, helps accelerate model development lifecycle by bringing both the transparency of your experiment and the reliability in your data, bridging the gap between science and engineering and enabling banks to have a more robust yet agile approach to risk management. This notebook uses stock data from Yahoo Finance (through `yfinance` library) for a synthetic Latin America portfolio enriched with news analytics from [GDELT](https://www.gdeltproject.org/).
# MAGIC
# MAGIC <img src="/files/antoine.amend/images/value_at_risk_workflow.png" alt="logical_flow" width="600">

# COMMAND ----------

# MAGIC %md
# MAGIC ## Value at risk 101
# MAGIC
# MAGIC VaR is measure of potential loss at a specific confidence interval. A VAR statistic has three components: a time period, a confidence level and a loss amount (or loss percentage). What is the most I can - with a 95% or 99% level of confidence - expect to lose in dollars over the next month? There are 3 ways to compute Value at risk
# MAGIC #
# MAGIC
# MAGIC + **Historical Method**: The historical method simply re-organizes actual historical returns, putting them in order from worst to best.
# MAGIC + **The Variance-Covariance Method**: This method assumes that stock returns are normally distributed and use pdf instead of actual returns.
# MAGIC + **Monte Carlo Simulation**: This method involves developing a model for future stock price returns and running multiple hypothetical trials.
# MAGIC
# MAGIC We report in below example a simple Value at risk calculation for a synthetic instrument, given a volatility (i.e. standard deviation of instrument returns) and a time horizon (300 days). **What is the most I could lose in 300 days with a 95% confidence?**

# COMMAND ----------

# DBTITLE 1,Control parameters
# time horizon
days = 300
dt = 1 / float(days)

# volatility
sigma = 0.04

# drift (average growth rate)
mu = 0.05

# initial starting price
start_price = 10

# number of simulations
runs_gr = 500
runs_mc = 10000

# COMMAND ----------

# DBTITLE 1,Plot monte-carlo simulations for 300 days return
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def generate_prices(start_price):
    shock = np.zeros(days)
    price = np.zeros(days)
    price[0] = start_price
    for i in range(1, days):
        shock[i] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        price[i] = max(0, price[i - 1] + shock[i] * price[i - 1])
    return price

plt.figure(figsize=(16,6))
for i in range(1, runs_gr):
    plt.plot(generate_prices(start_price))

plt.title('Simulated price')
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()

# COMMAND ----------

# DBTITLE 1,Compute value at risk
simulations = np.zeros(runs_mc)
for i in range(0, runs_mc):
    simulations[i] = generate_prices(start_price)[days - 1]

mean = simulations.mean()
z = stats.norm.ppf(1 - 0.95)
m1 = simulations.min()
m2 = simulations.max()
std = simulations.std()
q1 = np.percentile(simulations, 5)  # VAR95

x1 = np.arange(9, 12, 0.01)
y1 = stats.norm.pdf(x1, loc=mean, scale=std)
x2 = np.arange(x1.min(), q1, 0.001)
y2 = stats.norm.pdf(x2, loc=mean, scale=std)

mc_df = pd.DataFrame(data=simulations, columns=['return'])
ax = mc_df.hist(column='return', bins=50, density=True, grid=False, figsize=(12, 8), color='#86bf91', zorder=2,
                rwidth=0.9)
ax = ax[0]

for x in ax:
    x.spines['right'].set_visible(False)
    x.spines['top'].set_visible(False)
    x.spines['left'].set_visible(False)
    x.axvline(x=q1, color='r', linestyle='dashed', linewidth=1)
    x.fill_between(x2, y2, zorder=3, alpha=0.4)
    x.plot(x1, y1, zorder=3)
    x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off",
                  labelleft="on")
    vals = x.get_yticks()
    for tick in vals:
        x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    x.set_title("VAR95 = {:.3f}".format(q1), weight='bold', size=15)
    x.set_xlabel("{} days returns".format(days), labelpad=20, weight='bold', size=12)
    x.set_ylabel("Density", labelpad=20, weight='bold', size=12)
#    x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

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