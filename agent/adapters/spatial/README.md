# spatial adapter



```

  <skill-name>/SKILL.md

  Dockerfile / task.toml / workspace/*.py / tests/ / solution/
```



```bash
python adapters/spatial/run_adapter.py --output-dir tasks/erqa_disk_400 ...   # ERQA
python adapters/spatial/load_mmsi.py  --output-dir tasks/mmsi_disk_100 ...    # MMSI
```

（`--vision-mode` / `--skill-mode` / `--limit` / `--overwrite`）。



```bash
wc -c tasks/<job>/MMSI_0/instruction.md
```

