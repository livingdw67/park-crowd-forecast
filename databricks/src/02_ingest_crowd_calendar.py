# Databricks notebook source
# MAGIC %md
# MAGIC # 02 Ingest: queue-times crowd calendar into a bronze Delta table
# MAGIC
# MAGIC Scrapes the monthly crowd calendar pages from queue-times.com (Powered by
# MAGIC [Queue-Times.com](https://queue-times.com/)): a daily crowd score, park hours, weather and
# MAGIC holiday flags per park. This is the long history the forecast trains on; the live ingest only
# MAGIC starts on 2026-10-22.
# MAGIC
# MAGIC The first run (or `backfill=true`) walks every month from each park's first month. Later runs
# MAGIC refresh the last two months and the next two, since past days flip from forecast to measured
# MAGIC and upcoming park hours get published. Raw HTML is kept in the volume, and a backfill reuses
# MAGIC saved pages for months before that window instead of fetching them again.
# MAGIC Requests are 1.5 seconds apart.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "parks")
dbutils.widgets.text("backfill", "false")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
backfill = dbutils.widgets.get("backfill").lower() == "true"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.raw_files")
volume_dir = f"/Volumes/{catalog}/{schema}/raw_files/crowd_calendar"
table = f"{catalog}.{schema}.raw_crowd_calendar"

# First month with data on queue-times for each park
first_month = {
    65: (2018, 7),   # Universal Studios Florida
    64: (2018, 7),   # Islands of Adventure
    334: (2025, 5),  # Epic Universe (opened 2025-05-22)
    67: (2018, 7),   # Volcano Bay (months without data redirect elsewhere and parse to nothing)
}

# COMMAND ----------

import os
import time
import urllib.request
from datetime import date, datetime, timezone

from parks_lib.queue_times import USER_AGENT, parse_calendar_month

def add_months(year, month, n):
    i = year * 12 + month - 1 + n
    return i // 12, i % 12 + 1

today = date.today()
last_month = add_months(today.year, today.month, 2)
refresh_from = add_months(today.year, today.month, -2)
if not backfill and spark.catalog.tableExists(table):
    start = {park_id: refresh_from for park_id in first_month}
else:
    backfill = True
    start = first_month

def months_from(first):
    ym = first
    while ym <= last_month:
        yield ym
        ym = add_months(*ym, 1)

def fetch_page(park_id, year, month):
    url = f"https://queue-times.com/parks/{park_id}/calendar/{year}/{month:02d}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8")

# COMMAND ----------

scraped_at = datetime.now(timezone.utc)
rows, failures = [], []
for park_id, first in start.items():
    os.makedirs(f"{volume_dir}/park_{park_id}", exist_ok=True)
    for year, month in months_from(first):
        path = f"{volume_dir}/park_{park_id}/{year}-{month:02d}.html"
        # Months older than the refresh window don't change, so a saved page is reused
        if (year, month) < refresh_from and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                page = f.read()
        else:
            try:
                page = fetch_page(park_id, year, month)
            except Exception as e:
                failures.append((park_id, year, month, repr(e)))
                continue
            finally:
                time.sleep(1.5)
            with open(path, "w", encoding="utf-8") as out:
                out.write(page)
        rows += [{**day, "scraped_at": scraped_at} for day in parse_calendar_month(page, park_id, year, month)]

print(f"Mode: {'backfill' if backfill else 'refresh'}; days parsed: {len(rows):,}; failed pages: {len(failures)}")
for f in failures:
    print("  failed:", f)
assert rows, "No calendar days parsed"

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {table} (
        park_id INT, date DATE, is_closed BOOLEAN, crowd_percent INT, crowd_level STRING,
        is_forecast BOOLEAN, is_public_holiday BOOLEAN, weather_icons ARRAY<STRING>,
        open_time STRING, close_time STRING, extra_hours_open STRING, extra_hours_close STRING,
        scraped_at TIMESTAMP
    )
""")

spark.createDataFrame(rows, schema=spark.table(table).schema).createOrReplaceTempView("new_days")
spark.sql(f"""
    MERGE INTO {table} t
    USING new_days s
    ON t.park_id = s.park_id AND t.date = s.date
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")

display(spark.sql(f"""
    SELECT park_id, min(date) first_day, max(date) last_day, count(*) days,
           sum(CAST(is_closed AS INT)) closed, sum(CAST(is_forecast AS INT)) forecast,
           round(avg(CASE WHEN NOT is_forecast THEN crowd_percent END), 1) avg_measured_crowd
    FROM {table} GROUP BY park_id ORDER BY park_id
"""))
