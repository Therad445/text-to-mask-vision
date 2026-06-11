# Experiment 03: Batch Gallery Summary

This file summarizes the qualitative batch gallery run.

| Case | Prompt | Raw detections | Final detections | Note |
|---|---|---:|---:|---|
| `01_crowded_bears` | `bear .` | 8 | 5 | Crowded scene with five final detections; tuned setting balances recall and visualization cleanliness. |
| `02_bear_cub_prompt` | `bear cub .` | 5 | 4 | More specific prompt on the same bear image; useful for prompt sensitivity. |
| `03_bus_person_multi_object` | `bus . person .` | 5 | 5 | Multi-object prompt with bus and person detections. |
| `04_person_clear_case` | `person .` | 2 | 2 | Simple clear person case. |
| `05_weak_prompt_case` | `small animal .` | 5 | 5 | Mismatched prompt; useful as a limitation example. |

The crowded bear image is the most challenging case because it contains several overlapping animals.
The final tuned setting keeps five detections for five visible bears without adding an extra phantom detection.

The bus/person and weak-prompt cases intentionally use the same image to isolate prompt sensitivity.
