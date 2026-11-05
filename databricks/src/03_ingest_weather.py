# Databricks notebook source
# MAGIC %md
# MAGIC # 03 Ingest: daily Orlando weather into a bronze Delta table
# MAGIC
# MAGIC Weather from [Open-Meteo](https://open-meteo.com/) (CC BY 4.0). The archive API gives observed
# MAGIC history back to 2018 and the forecast API the next 16 days. Each day's row is replaced as better
# MAGIC data arrives, so a forecast day becomes observed once the archive covers it. Days past the
# MAGIC forecast horizon have no weather row; the model falls back to seasonal averages for those.
# MAGIC
# MAGIC The first run loads all history; later runs refetch only the last three months of the archive
# MAGIC (Open-Meteo counts long date ranges as many calls), so older rows stay as they are.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "parks")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
table = f"{catalog}.{schema}.raw_weather_daily"

# COMMAND ----------

from datetime import date, datetime, timedelta, timezone

import pandas as pd

from parks_lib.weather import ARCHIVE_URL, FORECAST_URL, archive_start, fetch_daily

today = date.today()
last_observed = (spark.sql(f"SELECT max(date) FROM {table} WHERE NOT is_forecast").first()[0]
                 if spark.catalog.tableExists(table) else None)
start = archive_start(last_observed, today)
print(f"Last observed day in the table: {last_observed}; fetching the archive from {start}")
observed = fetch_daily(ARCHIVE_URL, start_date=str(start), end_date=str(today - timedelta(days=1)))
observed["is_forecast"] = False
forecast = fetch_daily(FORECAST_URL, forecast_days=16)
forecast["is_forecast"] = True

# Observed wins where both exist; the archive's newest days can be empty while it catches up
observed = observed.dropna(subset=["temperature_2m_max"])
weather = pd.concat([observed, forecast[~forecast["time"].isin(observed["time"])]], ignore_index=True)
weather = weather.rename(columns={"time": "date"})
weather["date"] = pd.to_datetime(weather["date"]).dt.date
weather["weather_code"] = weather["weather_code"].astype("Int64")
weather["fetched_at"] = datetime.now(timezone.utc)
print(f"Days: {len(weather):,} ({(~weather['is_forecast']).sum():,} observed, {weather['is_forecast'].sum()} forecast), "
      f"{weather['date'].min()} to {weather['date'].max()}")

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {table} (
        date DATE, temperature_2m_max DOUBLE, temperature_2m_min DOUBLE, apparent_temperature_max DOUBLE,
        precipitation_sum DOUBLE, precipitation_hours DOUBLE, weather_code INT, is_forecast BOOLEAN,
        fetched_at TIMESTAMP
    )
""")
spark.createDataFrame(weather[spark.table(table).columns], schema=spark.table(table).schema).createOrReplaceTempView("new_weather")
spark.sql(f"""
    MERGE INTO {table} t
    USING new_weather s
    ON t.date = s.date
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")
print(f"Total rows in raw_weather_daily: {spark.table(table).count():,}")
