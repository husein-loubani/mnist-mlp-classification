# Data dictionary

## Source

Kaggle [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer), `train.csv`, downloaded through the Kaggle API and cached to `data/raw/`.

The competition also ships `test.csv` with 28,000 images. It carries no labels because it scores the public leaderboard, so it is not used here: every split comes out of the labeled file.

## Columns

| Column | Type | Description |
|---|---|---|
| `label` | int | The digit shown, 0 to 9 |
| `pixel0` … `pixel783` | int | Grayscale intensity, 0 (black) to 255 (white) |

The 784 pixel columns are a 28x28 image flattened in row-major order, so `pixel{r*28+c}` is the pixel at row `r`, column `c`.

## Shape

| Property | Value |
|---|---|
| Images | 42,000 |
| Pixels per image | 784 (28 x 28) |
| Classes | 10, digits 0 to 9 |
| Class balance | 9.04% (digit 5) to 11.15% (digit 1) |

## Derived values

| Name | Definition | Why it exists |
|---|---|---|
| ink pixels | count of pixels above zero | Detects blank and near-blank frames during cleaning |
| normalized pixels | `(pixel / 255 - mean) / std` | Model input; mean and std come from the **training split only** |
| class weights | inverse class frequency | Available for imbalance; measured as unnecessary on this data |

## Cleaning rules

| Rule | Reason |
|---|---|
| Drop duplicate images, matched on pixels alone | The same picture with two labels is a contradiction; both copies go |
| Drop blank and near-blank frames | Fewer than 10 ink pixels is not a digit |
| Drop pixels outside 0–255 | Out of range means corrupted |
| Drop labels outside 0–9 | Not a valid class |

All four ran against this file and removed nothing, which the audit predicted. The rules are exercised against deliberately broken frames in `tests/test_dataset.py`.

**Ordering matters:** cleaning runs before the split. Deduplicating afterwards would leave copies of the same image on both sides of the boundary.
