# Error Analysis

## Worst-performing classes (by F1)

|                     |   precision |   recall |   f1-score |   support |
|:--------------------|------------:|---------:|-----------:|----------:|
| Lupus               |    0.642857 | 0.529412 |   0.580645 |        34 |
| Moles               |    0.585366 | 0.6      |   0.592593 |        40 |
| Infestations_Bites  |    0.783784 | 0.483333 |   0.597938 |        60 |
| Sun_Sunlight_Damage |    0.594595 | 0.647059 |   0.619718 |        34 |
| Vascular_Tumors     |    0.576923 | 0.75     |   0.652174 |        60 |

## Most confused class pairs

| true_class         | predicted_as    |   count |
|:-------------------|:----------------|--------:|
| SkinCancer         | Benign_tumors   |      13 |
| Vascular_Tumors    | Benign_tumors   |       9 |
| Eczema             | Tinea           |       8 |
| Eczema             | Psoriasis       |       7 |
| Moles              | Benign_tumors   |       7 |
| Unknown_Normal     | Vitiligo        |       7 |
| Infestations_Bites | Benign_tumors   |       6 |
| Seborrh_Keratoses  | Benign_tumors   |       5 |
| Psoriasis          | Eczema          |       5 |
| Benign_tumors      | Vascular_Tumors |       5 |

## High-confidence wrong predictions

378 total wrong predictions. Top 20 by confidence shown in `high_confidence_errors.csv`. Mean confidence on wrong predictions: 0.722

## Low-confidence correct predictions

Mean confidence on correct predictions: 0.941. Bottom 20 by confidence shown in `low_confidence_correct.csv`.

## Summary

- Total predictions: 1546
- Correct: 1168 (75.5%)
- Wrong: 378 (24.5%)
- Worst class by F1: Lupus (F1=0.581)
- Most confused pair: 'SkinCancer' predicted as 'Benign_tumors' (13 times)
