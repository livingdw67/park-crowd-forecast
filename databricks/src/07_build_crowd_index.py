# Databricks notebook source
# MAGIC %md
# MAGIC # 07 Crowd index: our own daily crowd measure from ride wait times
# MAGIC
# MAGIC The forecast trains on queue-times' crowd score, whose formula isn't published. This builds a
# MAGIC transparent alternative from the hourly wait-time readings:
# MAGIC
# MAGIC - `wait_times_clean` (silver): readings in park-local time, minus stale readings (more than an
# MAGIC   hour older than the fetch), single-rider lines and event attractions (Halloween Horror Nights
# MAGIC   houses arrive under the land "Halloween").
# MAGIC - `daily_crowd_index` (gold): per park and day, the average posted wait across open rides
# MAGIC   between 11am and 5pm, plus how many rides and readings back it. Joined to queue-times' score so
# MAGIC   the two can be compared as days accumulate.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "parks")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
spark.sql(f"USE {catalog}.{schema}")

# COMMAND ----------

spark.sql("""
    CREATE OR REPLACE TABLE wait_times_clean AS
    WITH local AS (
        SELECT *,
               from_utc_timestamp(last_updated, 'America/New_York') AS local_time,
               timestampdiff(MINUTE, last_updated, fetched_at) AS age_minutes
        FROM raw_wait_times
    )
    SELECT park_id, park_name, land_name, ride_id, ride_name, is_open, wait_time,
           local_time, to_date(local_time) AS local_date, hour(local_time) AS local_hour
    FROM local
    WHERE age_minutes <= 60
      AND ride_name NOT ILIKE '%single rider%'
      AND coalesce(land_name, '') NOT IN ('Halloween')
""")

# COMMAND ----------

spark.sql("""
    CREATE OR REPLACE TABLE daily_crowd_index AS
    WITH midday AS (
        SELECT park_id, local_date, ride_id, avg(wait_time) AS ride_avg_wait, count(*) AS readings
        FROM wait_times_clean
        WHERE is_open AND wait_time > 0 AND local_hour BETWEEN 11 AND 16
        GROUP BY park_id, local_date, ride_id
    )
    SELECT m.park_id, m.local_date AS date,
           round(avg(m.ride_avg_wait), 1) AS avg_wait_minutes,
           round(percentile_approx(m.ride_avg_wait, 0.9), 1) AS p90_ride_wait_minutes,
           count(*) AS rides, sum(m.readings) AS readings,
           c.crowd_percent AS queue_times_crowd_percent
    FROM midday m
    LEFT JOIN raw_crowd_calendar c ON c.park_id = m.park_id AND c.date = m.local_date AND NOT c.is_forecast
    GROUP BY m.park_id, m.local_date, c.crowd_percent
""")
display(spark.sql("SELECT * FROM daily_crowd_index ORDER BY date DESC, park_id"))
