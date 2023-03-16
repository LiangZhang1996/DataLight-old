## Introduction

Thses code are used to generate offline datasets.
- For `Cycle`, use `run_fixedtime.py`
- For `Expert`, use `run_advanced_colight.py`
- For `Random`, use `run_random.py`

## Steps

1. Run the abovementioned files to generate the memeories.
2. Organize the memories into new folders.
3. Use `organize_memory.py` to generate the offline datasets (`xx.pkl`).
4. Move the `xx.pkl` into `DataLight/memory`.

For step-2, the memories(`total_samoles_inter_x.pkl`) are moved into the flowing paths:
- ./JN/1/
- ./JN/2
- ./JN/3
- ./HZ/1/
- ./HZ/2/