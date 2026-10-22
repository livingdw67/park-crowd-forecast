# Databricks notebook source
# MAGIC %md
# MAGIC # 03 Ingest: daily Orlando weather into a bronze Delta table
# MAGIC
# MAGIC Weather from [Open-Meteo](https://open-meteo.com/) (CC BY 4.0). The archive API gives observed
# MAGIC history back to 2018 and the forecast API the next 16 days. Each day's row is replaced as better
# MAGIC data arrives, so a forecast day becomes observed once the archive covers it. Days past the
# MAGIC forecast horizon have no weather row; the model falls back to seasonal averages for those.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "parks")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
table = f"{catalog}.{schema}.raw_weather_daily"

# COMMAND ----------

import json
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

import pandas as pd

# Universal Orlando Resort
LATITUDE, LONGITUDE = 28.47, -81.47
DAILY = ["temperature_2m_max", "temperature_2m_min", "apparent_temperature_max",
         "precipitation_sum", "precipitation_hours", "weather_code"]

def fetch_daily(base_url, **params):
    query = urllib.parse.urlencode({
        "latitude": LATITUDE, "longitude": LONGITUDE, "daily": ",".join(DAILY),
        "temperature_unit": "fahrenheit", "precipitation_unit": "inch", "timezone": "America/New_York", **params,
    })
    with urllib.request.urlopen(f"{base_url}?{query}", timeout=60) as response:
        return pd.DataFrame(json.loads(response.read())["daily"])

today = date.today()
observed = fetch_daily("https://archive-api.open-meteo.com/v1/archive", start_date="2018-01-01", end_date=str(today - timedelta(days=1)))
observed["is_forecast"] = False
forecast = fetch_daily("https://api.open-meteo.com/v1/forecast", forecast_days=16)
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
