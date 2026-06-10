# Experiment 04: Evaluation Summary

This summary aggregates the qualitative batch gallery experiment.
The project does not use a manually labeled benchmark, so the evaluation combines visual inspection with proxy metrics from the detection and segmentation pipeline.

| Case | Prompt | Raw detections | After NMS | Mean box score | Mean mask score | Interpretation |
|---|---|---:|---:|---:|---:|---|
| `01_crowded_bears` | `bear .` | 8 | 4 | 0.438 | 0.951 | Crowded scene; NMS reduces duplicate/overlapping detections. |
| `02_bear_cub_prompt` | `bear cub .` | 5 | 4 | 0.425 | 0.958 | Same image with a more specific prompt; useful for prompt sensitivity. |
| `03_bus_person_multi_object` | `bus . person .` | 5 | 5 | 0.709 | 0.962 | Successful multi-object prompt with bus and person categories. |
| `04_person_clear_case` | `person .` | 2 | 2 | 0.654 | 0.987 | Clear simple case with stable person detections. |
| `05_weak_prompt_case` | `small animal .` | 5 | 5 | 0.286 | 0.961 | Mismatched prompt; lower box scores show semantic uncertainty. |

Key observations:

- The multi-object `bus . person .` prompt has the strongest mean box confidence among the gallery cases.
- The weak `small animal .` prompt produces much lower box confidence, which supports the prompt-sensitivity analysis.
- SAM mask scores remain high even in weak prompt cases, because SAM segments boxes provided by the detector; therefore detection quality is the main bottleneck.
- In crowded scenes, NMS reduces overlapping detections but may also remove valid nearby objects.
