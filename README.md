# Prompt Engineering для анализа сентимента

Домашняя работа: классификация тональности русскоязычных комментариев (3 класса)
инструктивной моделью **Qwen/Qwen2.5-1.5B-Instruct** без дообучения.

Основной артефакт — **`hw_prompt_sentiment.ipynb`**: выполненный ноутбук с кодом, графиками
и markdown-анализом по всем пунктам задания (данные → базовый prompt + greedy + постобработка →
вариации промпта → system prompts → стратегии и параметры генерации → итоги).

## Главные результаты (300 текстов, по 100 на класс)

| Эксперимент | Лучший результат | Контроль |
|---|---|---|
| п.2 Базовый prompt (greedy) | macro-F1 0.603 (parse 100%) | baselines 0.333 |
| п.3 Вариации промпта | **0.824** — нумерованные правила (p3) | свободная формулировка 0.207, few-shot 0.458 |
| п.4 System prompts | 0.824 — p3 **без** system; sys2 вытягивает слабые промпты: 0.675 → 0.775 | — |
| п.5 Стратегии генерации | **0.831** — beam3 (3× медленнее); greedy 0.824; t=1.0 0.776 | self-consistency n=5: 0.826 |

Вывод: формулировка промпта (+0.22 macro-F1) ≫ system prompt (до +0.11, только для слабых промптов)
≫ стратегия генерации (≤ +0.007). Остаточные ошибки — нейтральный класс (ирония/шум разметки).

## Воспроизведение

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name hw-prompt --display-name hw-prompt

# пересобрать «чистый» ноутбук из scripts/build_notebook.py и выполнить его целиком
.venv/bin/python scripts/build_notebook.py
.venv/bin/python scripts/run_notebook.py hw_prompt_sentiment.ipynb
.venv/bin/python scripts/fill_analysis.py   # вписать аналитические markdown-ячейки по фактам
```

Датасет `sy-volkov/russian-sentiment-analysis-dataset` и модель `Qwen/Qwen2.5-1.5B-Instruct`
скачиваются автоматически из Hugging Face Hub. Полное выполнение на Apple M1 Max (mps, bfloat16)
занимает ~36 минут (~30 конфигураций × 300 текстов).

## Структура

- `hw_prompt_sentiment.ipynb` — основной артефакт
- `scripts/build_notebook.py` — генератор ноутбука
- `scripts/run_notebook.py` — исполнитель с сохранением после каждой ячейки
- `scripts/fill_analysis.py` — вставка аналитических выводов после прогона
- `scripts/smoke_test.py` — быстрая проверка API генерации до полного запуска
- `logs/metrics.jsonl` — метрики всех конфигураций (`data/`, `outputs/`, `logs/` в `.gitignore`)

## Методологические заметки

- Дефолтный `generation_config` Qwen включает сэмплирование (`do_sample=True`, t=0.7,
  top_p=0.8, top_k=20, repetition_penalty=1.1) — для greedy все параметры задаются явно.
- ChatML-шаблон Qwen сам подставляет дефолтный system prompt, когда system-сообщения нет:
  «без system prompt» в экспериментах = «с дефолтным системным промптом Qwen».
- Оценка — стратифицированная сбалансированная подвыборка (3×100); нераспознанные ответы
  относятся к классу 0, доля распознанных (`parse_rate`) — отдельная метрика.
