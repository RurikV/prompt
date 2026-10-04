"""Разовый патч: обновляет plot_confusion в выполненном ноутбуке и перезапускает
только префикс ячеек (до confusion matrix включительно), не трогая остальные выходы.

Запуск из корня проекта:
    .venv/bin/python scripts/patch_confusion.py
"""
import shutil
import subprocess
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB = "hw_prompt_sentiment.ipynb"
shutil.copy(NB, "logs/nb_executed_backup.ipynb")
shutil.copy("logs/metrics.jsonl", "logs/metrics.jsonl.bak")

# 1) пересобираем «чистый» ноутбук с исправленным plot_confusion (перезапишет NB — бэкап снят)
subprocess.run([".venv/bin/python", "scripts/build_notebook.py"], check=True, cwd=".")
fresh = nbformat.read(NB, as_version=4)

# 2) возвращаем выполненный ноутбук и вшиваем в него исправленный источник ячейки-хелпера
nb = nbformat.read("logs/nb_executed_backup.ipynb", as_version=4)
fresh_helper = next(c.source for c in fresh.cells if "def plot_confusion" in (c.source or ""))
for c in nb.cells:
    if c.cell_type == "code" and "def plot_confusion" in c.source:
        c.source = fresh_helper
        break

# 3) исполняем префикс до ячейки с BASE_PROMPT включительно в свежем кернеле
cut = next(i for i, c in enumerate(nb.cells)
           if c.cell_type == "code" and c.source.startswith("BASE_PROMPT")) + 1
sub = nbformat.v4.new_notebook(cells=nb.cells[:cut], metadata=nb.metadata)
NotebookClient(sub, timeout=900, kernel_name="hw-prompt",
               resources={"metadata": {"path": str(Path.cwd())}}).execute()

# 4) обновлённые ячейки уже в объекте nb (sub делит те же объекты ячеек) — сохраняем
nbformat.write(nb, NB)

# 5) лог метрики восстанавливаем: перезапуск префикса сбросил файл и дописал только 2_base
shutil.copy("logs/metrics.jsonl.bak", "logs/metrics.jsonl")
print(f"патч применён, перезапущено ячеек: {cut}, метрики восстановлены")
