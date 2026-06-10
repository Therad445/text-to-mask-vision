# Experiment 03 — Result Gallery and Prompt Sensitivity

## Goal

The goal of this experiment is to test the text-to-mask pipeline on several images and prompts, including both successful and difficult cases.

## Planned Cases

| Case | Image Type | Prompt | Expected Behavior | Purpose |
|---|---|---|---|---|
| Good case 1 | Single clear object | `dog .` | One accurate box and mask | Basic success case |
| Good case 2 | Multiple objects | `person . car .` | Several boxes and masks | Multi-object prompt |
| Crowded case | Overlapping bears | `bear .` | Partial overlap and duplicate boxes | Error analysis |
| Prompt sensitivity | Same image, different prompts | `bear .` vs `bear cub .` | Different detections | Prompt analysis |
| Weak/no detection | Ambiguous or small object | custom prompt | Low confidence or no detection | Limitation case |

## Metrics / Recorded Values

For each case, record:

- prompt;
- box threshold;
- text threshold;
- NMS IoU threshold;
- number of raw detections;
- number of detections after NMS;
- box scores;
- mask scores;
- qualitative observation.

## Expected Outcome

The gallery should show that the pipeline works on clear cases, but also has understandable limitations in crowded scenes, occlusion, small objects, and prompt-sensitive cases.

## Notes

This experiment is qualitative. The goal is not to claim benchmark-level accuracy, but to demonstrate a working open-vocabulary detection and segmentation system and analyze its behavior.
