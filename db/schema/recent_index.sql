-- Schema for RecentIndex and MemoryService
CREATE TABLE IF NOT EXISTS recent_index (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  phash TEXT NOT NULL UNIQUE,
  canonical_uri TEXT NOT NULL,
  last_seen INTEGER NOT NULL,
  hit_count INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_recent_index_phash ON recent_index(phash);

CREATE TABLE IF NOT EXISTS memory_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT,
  timestamp INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_memory_events_runid ON memory_events(run_id);
