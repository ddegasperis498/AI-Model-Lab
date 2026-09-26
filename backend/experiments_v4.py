from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List


class ExperimentStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self):
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def _init(self):
        with self._connect() as con:
            con.execute('''CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                notes TEXT NOT NULL,
                config_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL
            )''')

    def save(self, name: str, kind: str, config: Dict[str, Any], metrics: Dict[str, Any], notes: str = '') -> int:
        with self._connect() as con:
            cur = con.execute(
                'INSERT INTO experiments(created_at,name,kind,notes,config_json,metrics_json) VALUES(?,?,?,?,?,?)',
                (time.time(), name, kind, notes, json.dumps(config, ensure_ascii=False), json.dumps(metrics, ensure_ascii=False)),
            )
            return int(cur.lastrowid)

    def list(self, limit: int = 100) -> List[dict]:
        with self._connect() as con:
            rows = con.execute('SELECT * FROM experiments ORDER BY id DESC LIMIT ?', (int(limit),)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d['config'] = json.loads(d.pop('config_json'))
            d['metrics'] = json.loads(d.pop('metrics_json'))
            out.append(d)
        return out

    def delete(self, experiment_id: int):
        with self._connect() as con:
            con.execute('DELETE FROM experiments WHERE id=?', (int(experiment_id),))
