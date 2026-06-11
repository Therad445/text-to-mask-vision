# Text-to-Mask Vision

## 1. Титульный слайд
Text-to-Mask Vision: изображение + prompt → boxes → masks → demo.

## 2. Карта защиты
Проблема → пайплайн → реализация → эксперименты → ограничения.

## 3. Зачем нужен проект
Fixed-class detectors неудобны для быстрых запросов; text-to-mask даёт интерактивность.

## 4. Что я проверял
Можно ли собрать practical pipeline без обучения новой модели под каждый класс.

## 5. Архитектура
Grounding DINO находит boxes, NMS убирает дубли, SAM строит masks.

## 6. Что реализовано
Reusable pipeline, Streamlit demo, Colab experiments, CSV/results, report, checks.

## 7. Demo
Локальное web-приложение: image upload, prompt, thresholds, NMS, detections table.

## 8. Дизайн экспериментов
5 кейсов: success, prompt sensitivity, weak prompt, crowded failure analysis.

## 9. Success case
`bus . person .`: mean box score 0.709, mean mask score 0.962.

## 10. Prompt sensitivity
Одна картинка, разные prompts: `bus . person .` vs `small animal .`.

## 11. Crowded bears
Финальный preset: box=0.25, text=0.30, NMS=0.35, max=8.
8 raw detections → 5 final detections.
Mean box score 0.399, mean mask score 0.971.

## 12. Что получилось и что ограничено
Получилось: рабочий pipeline и demo.
Ограничения: нет labeled benchmark, crowded scenes, prompt sensitivity, latency.

## 13. Финал
Главный результат: воспроизводимая text-to-mask система на Grounding DINO + SAM.
