# Experiment 03 — Result Gallery and Prompt Sensitivity

## Goal

This experiment evaluates the text-to-mask pipeline on several qualitative cases:

- crowded scenes;
- multi-object prompts;
- clear object cases;
- prompt sensitivity;
- weak or mismatched prompts.

## Results

| Case | Prompt | Raw detections | After NMS | Notes |
|---|---|---:|---:|---|
| 01_crowded_bears | `bear .` | 8 | 4 | Crowded scene with visually overlapping bear instances. |
| 02_bear_cub_prompt | `bear cub .` | 5 | 4 | Prompt sensitivity case on the same image. |
| 03_bus_person_multi_object | `bus . person .` | 5 | 5 | Multi-object prompt with a bus and people. |
| 04_person_clear_case | `person .` | 2 | 2 | Clear person detection case. |
| 05_weak_prompt_case | `small animal .` | 5 | 5 | Weak or mismatched prompt case. |

## Observation

The pipeline works well on clear objects and multi-object prompts. Crowded scenes remain challenging because Grounding DINO can produce duplicate or overlapping boxes, and NMS can suppress nearby valid detections.

The weak prompt case illustrates that open-vocabulary detection is sensitive to prompt wording.