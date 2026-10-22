# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Ingest: live ride wait times into a bronze Delta table
# MAGIC
# MAGIC Pulls the current wait time for every ride in each park from the queue-times.com API
# MAGIC (Powered by [Queue-Times.com](https://queue-times.com/)). The raw JSON goes to a Unity Catalog
# MAGIC volume so any snapshot can be replayed, and the parsed rows are merged into `raw_wait_times`.
# MAGIC Runs hourly; a ride's reading is keyed by `last_updated`, so a reading that hasn't changed
# MAGIC since the last run isn't stored twice.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "parks")
dbutils.widgets.text("park_ids", "65,64,334,67")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
park_ids = [int(p) for p in dbutils.widgets.get("park_ids").split(",")]

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.raw_files")
volume_dir = f"/Volumes/{catalog}/{schema}/raw_files/queue_times"

# COMMAND ----------

import json
import os
import urllib.request
from datetime import datetime, timezone

def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "park-crowd-forecast (portfolio project)"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read())

park_names = {p["id"]: p["name"] for group in fetch_json("https://queue-times.com/parks.json") for p in group["parks"]}

fetched_at = datetime.now(timezone.utc)
snapshots = {}
for park_id in park_ids:
    payload = fetch_json(f"https://queue-times.com/parks/{park_id}/queue_times.json")
    snapshots[park_id] = payload
    park_dir = f"{volume_dir}/park_{park_id}/{fetched_at:%Y-%m-%d}"
    os.makedirs(park_dir, exist_ok=True)
    with open(f"{park_dir}/{fetched_at:%H%M%S}.json", "w") as out:
        json.dump(payload, out)

# COMMAND ----------

# Rides sit under "lands" in most parks, but some parks list them in a flat top-level "rides" array
def parse(park_id, payload):
    groups = [(land["name"], land["rides"]) for land in payload.get("lands", [])]
    groups.append((None, payload.get("rides", [])))
    return [
        {
            "park_id": park_id,
            "park_name": park_names.get(park_id),
            "land_name": land_name,
            "ride_id": ride["id"],
            "ride_name": ride["name"],
            "is_open": ride["is_open"],
            "wait_time": ride["wait_time"],
            "last_updated": datetime.fromisoformat(ride["last_updated"].replace("Z", "+00:00")),
            "fetched_at": fetched_at,
        }
        for land_name, rides in groups
        for ride in rides
    ]

rows = [row for park_id, payload in snapshots.items() for row in parse(park_id, payload)]
print(f"Rides fetched: {len(rows)} across {len(snapshots)} parks")

# COMMAND ----------

table = f"{catalog}.{schema}.raw_wait_times"
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {table} (
        park_id INT, park_name STRING, land_name STRING, ride_id INT, ride_name STRING,
        is_open BOOLEAN, wait_time INT, last_updated TIMESTAMP, fetched_at TIMESTAMP
    )
""")

schema_ddl = spark.table(table).schema
spark.createDataFrame(rows, schema=schema_ddl).createOrReplaceTempView("new_readings")
spark.sql(f"""
    MERGE INTO {table} t
    USING new_readings s
    ON t.ride_id = s.ride_id AND t.last_updated = s.last_updated
    WHEN NOT MATCHED THEN INSERT *
""")
print(f"Total rows in raw_wait_times: {spark.table(table).count():,}")
