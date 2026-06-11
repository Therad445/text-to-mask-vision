# Speaker notes — Text-to-Mask Vision

## 1. Title
Здравствуйте! Я представляю проект Text-to-Mask Vision — систему, которая по изображению и текстовому запросу строит bounding boxes и segmentation masks.

Главная идея: `image + text prompt -> boxes -> masks`. В проекте я объединяю Grounding DINO для open-vocabulary detection и SAM для box-guided segmentation.

## 2. Why this project matters
Классические детекторы обычно работают с фиксированным набором классов. Это неудобно, если пользователь хочет быстро искать произвольный объект через текст.

В моём проекте пользователь пишет prompt, например `bear .` или `bus . person .`, и система пытается найти соответствующие объекты без дообучения новой модели.

## 3. End-to-end pipeline
Pipeline состоит из шагов: входное изображение и prompt, нормализация prompt, Grounding DINO, NMS, SAM и финальная визуализация с таблицей detections.

Важный момент: Grounding DINO отвечает за локализацию, а SAM строит маску по уже выбранному box. Поэтому если box неточный, mask тоже может быть частичной.

## 4. Models used
Grounding DINO используется для open-vocabulary object detection: на вход получает изображение и текст, на выходе даёт boxes, scores и labels.

SAM используется для segmentation: он получает image + selected boxes и возвращает masks и mask scores.

Я не обучал новую модель с нуля. Цель была сделать практическую интеграцию pretrained foundation models в рабочую систему.

## 5. Implementation and repository
В проекте есть reusable pipeline в `src/text_to_mask_pipeline.py`, Streamlit demo в `app.py`, sanity notebook и batch gallery notebook.

Большие веса не хранятся в GitHub: SAM checkpoint скачивается отдельно через script. Также есть `check_project.sh`, который проверяет структуру проекта и отсутствие больших локальных файлов в Git.

## 6. Interactive demo
Streamlit demo позволяет загрузить изображение, ввести prompt, настроить thresholds, NMS IoU и max detections.

На выходе пользователь видит result visualization и detections table с scores и координатами boxes.

## 7. Experiment design
Я сделал три экспериментальных блока: sanity check, local demo и batch gallery.

Batch gallery содержит пять кейсов: crowded bears, bear cub prompt, bus/person multi-object prompt, clear person case и weak prompt case. Это не только success examples, но и failure analysis.

## 8. Success case: multi-object prompt
Здесь prompt `bus . person .` показывает, что система может обрабатывать несколько категорий в одном запросе.

Это хороший пример open-vocabulary control: модель находит и bus, и несколько person instances, а SAM строит masks по найденным boxes.

## 9. Prompt sensitivity
Здесь специально используется одно и то же изображение для двух prompts: `bus . person .` и `small animal .`.

Это controlled comparison: изображение фиксировано, меняется только текстовый prompt. Корректный prompt даёт осмысленный результат, а слабый prompt приводит к семантически неправильным detections и более низкой уверенности.

## 10. Crowded bears
Это самый сложный кейс. На изображении пять видимых медведей, но они маленькие, близко расположены и перекрываются.

Я подобрал параметры так, чтобы получить пять финальных detections без лишнего шестого фантомного объекта: `box=0.25`, `text=0.30`, `NMS=0.35`, `max=8`.

Mean box score равен 0.399 — это умеренно, но нормально для crowded scene. Mean mask score 0.971 — высокий. Значит основной bottleneck здесь detection/localization, а не SAM.

## 11. Proxy metrics summary
Так как нет вручную размеченного benchmark dataset с ground-truth masks, я использую qualitative evaluation и proxy metrics: число detections after NMS, mean box score, mean mask score и visual inspection.

Это честная оценка для practical prototype. Для будущей работы можно добавить labeled benchmark и считать IoU, precision, recall или mAP.

## 12. Individual contribution
Мой вклад: собрал end-to-end pipeline, интегрировал Grounding DINO и SAM, добавил NMS, сделал Streamlit demo, Colab experiments, batch gallery, CSV results, report и presentation.

Проект не про обучение новой модели, а про практическую интеграцию foundation models и анализ их поведения.

## 13. Conclusion
Итог: проект реализует рабочую text-to-mask систему. Она подходит для interactive annotation, visual search и rapid prototyping.

Главные ограничения: prompt sensitivity, crowded scenes, slow CPU inference и зависимость SAM masks от качества boxes.

## Likely questions and answers

### Why no IoU / mAP?
Because this is a practical prototype without a manually labeled segmentation benchmark. I used qualitative evaluation and proxy metrics. Adding labeled ground truth and IoU/mAP is a natural future work direction.

### Why is the crowded bears box score only 0.399?
Because it is intentionally a difficult crowded scene: small overlapping instances of the same class. The goal was not to maximize confidence by detecting only easy objects, but to keep five detections for five visible bears without adding a false sixth detection.

### Why is mask score high even when boxes are imperfect?
SAM confidence is about mask generation for the given box prompt. If the detector provides an imperfect box, SAM can still confidently segment that box region. Therefore the bottleneck is detection/localization.

### Why use the same bus image for the weak prompt?
It is intentional: the image is fixed and only the prompt changes. This isolates prompt sensitivity.

### What is your contribution if models are pretrained?
The contribution is system integration, reusable pipeline, NMS post-processing, Streamlit UI, reproducible Colab experiments, gallery results, error analysis, report and final defense artifacts.
