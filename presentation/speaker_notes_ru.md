# Speaker notes — human defense version

## 1. Title
Я представляю Text-to-Mask Vision — систему, которая по изображению и текстовому prompt строит boxes и masks.

## 2. Карта защиты
Сначала объясню проблему фиксированных классов, потом архитектуру, реализацию, эксперименты и ограничения.

## 3. Зачем нужен проект
Обычные detectors ограничены заранее заданными классами. Мне было интересно проверить, можно ли сделать более интерактивный pipeline через natural-language prompt.

## 4. Что я проверял
Это не проект про обучение новой модели с нуля. Цель — собрать practical application из pretrained foundation models и понять, где оно работает, а где ломается.

## 5. Архитектура
Grounding DINO отвечает за localization: boxes and scores. Потом NMS убирает дубли. SAM получает boxes и строит masks. Поэтому главный bottleneck — качество detection/localization.

## 6. Реализация
Я вынес core logic в `src/text_to_mask_pipeline.py`, сделал Streamlit demo, Colab sanity check, batch gallery, CSV results, report, presentation и `check_project.sh`.

## 7. Demo
Demo позволяет загрузить image, ввести prompt и менять thresholds. Финальные default values выставлены под crowded-bears defense preset: box 0.25, text 0.30, NMS 0.35, max 8.

## 8. Эксперименты
Я взял 5 кейсов: успешный multi-object prompt, простой person case, prompt sensitivity и crowded bears как сложный failure-analysis.

## 9. Success case
`bus . person .` — хороший пример: prompt совпадает с объектами, mean box score высокий, masks стабильные.

## 10. Prompt sensitivity
Я специально использую одну и ту же bus image для правильного и неправильного prompt. Так видно, что open-vocabulary модели гибкие, но чувствительны к формулировке.

## 11. Crowded bears
Это главный сложный пример. На картинке 5 видимых медведей. Я подобрал параметры так, чтобы получить 5 final detections без ложного шестого объекта. Box score 0.399 умеренный из-за overlap, mask score 0.971 высокий, потому что SAM хорошо сегментирует выбранные boxes.

## 12. Что получилось
Получилось собрать рабочий reproducible pipeline и показать успехи и ограничения. Ограничения честные: нет labeled benchmark, crowded scenes сложны, prompt влияет на результат, latency/GPU deployment можно улучшать.

## 13. Финал
Главная мысль: проект показывает границу между красивым demo foundation models и реальной инженерной надёжностью.
