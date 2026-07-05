# Professional Evaluation Report: Marius Hills

- **Evaluated Tiles**: 3115
- **Decision Threshold**: 0.75
- **Bootstrap Confidence**: 0.95 (1000 resamples)

## Global & Per-Class Metrics

| model          |   ('apollo_site', 'Dice') |   ('apollo_site', 'F1') |   ('apollo_site', 'IoU') |   ('apollo_site', 'Precision') |   ('apollo_site', 'Recall') |   ('impact_crater', 'Dice') |   ('impact_crater', 'F1') |   ('impact_crater', 'IoU') |   ('impact_crater', 'Precision') |   ('impact_crater', 'Recall') |   ('lobate_scarp', 'Dice') |   ('lobate_scarp', 'F1') |   ('lobate_scarp', 'IoU') |   ('lobate_scarp', 'Precision') |   ('lobate_scarp', 'Recall') |   ('pit_skylight', 'Dice') |   ('pit_skylight', 'F1') |   ('pit_skylight', 'IoU') |   ('pit_skylight', 'Precision') |   ('pit_skylight', 'Recall') |   ('wrinkle_ridge', 'Dice') |   ('wrinkle_ridge', 'F1') |   ('wrinkle_ridge', 'IoU') |   ('wrinkle_ridge', 'Precision') |   ('wrinkle_ridge', 'Recall') |
|:---------------|--------------------------:|------------------------:|-------------------------:|-------------------------------:|----------------------------:|----------------------------:|--------------------------:|---------------------------:|---------------------------------:|------------------------------:|---------------------------:|-------------------------:|--------------------------:|--------------------------------:|-----------------------------:|---------------------------:|-------------------------:|--------------------------:|--------------------------------:|-----------------------------:|----------------------------:|--------------------------:|---------------------------:|---------------------------------:|------------------------------:|
| MaskRCNN-Best  |                    0.0000 |                  0.0000 |                   0.0000 |                         1.0000 |                      0.0000 |                      0.7252 |                    0.7307 |                     0.6716 |                           0.9140 |                        0.7228 |                     0.0000 |                   0.0000 |                    0.0000 |                          1.0000 |                       0.0000 |                     0.0000 |                   0.0000 |                    0.0000 |                          1.0000 |                       0.0000 |                      0.0459 |                    0.0459 |                     0.0305 |                           0.9507 |                        0.0456 |
| Panoptic-Best  |                    0.0000 |                  0.0000 |                   0.0000 |                         1.0000 |                      0.0000 |                      0.9091 |                    0.9091 |                     0.8781 |                           0.9037 |                        0.9700 |                     0.0000 |                   0.0000 |                    0.0000 |                          1.0000 |                       0.0000 |                     0.0000 |                   0.0000 |                    0.0000 |                          0.9091 |                       0.0909 |                      0.0130 |                    0.0584 |                     0.0068 |                           0.6571 |                        0.2058 |
| SmallUNet-Best |                    0.0000 |                  0.0000 |                   0.0000 |                         1.0000 |                      0.0000 |                      0.5580 |                    0.5619 |                     0.5113 |                           0.9127 |                        0.5492 |                     0.0000 |                   0.0000 |                    0.0000 |                          1.0000 |                       0.0000 |                     0.0000 |                   0.0000 |                    0.0000 |                          1.0000 |                       0.0000 |                      0.0119 |                    0.1895 |                     0.0063 |                           0.5687 |                        0.1448 |

## Pairwise Statistical Significance (Wilcoxon Signed-Rank Test)

Tests the null hypothesis that the metric distribution medians are equal. Significant differences ($p < 0.05$) are highlighted.

### SmallUNet-Best vs MaskRCNN-Best

| class                |      statistic |      p_value | significant_005   |
|:---------------------|---------------:|-------------:|:------------------|
| impact_crater        |    1.92306e+06 |   5.4498e-23 | True              |
| pit_skylight         |  nan           | nan          | False             |
| wrinkle_ridge        | 1413           |   0.00199965 | True              |
| lobate_scarp         |  nan           | nan          | False             |
| irregular_mare_patch |  nan           | nan          | False             |
| apollo_site          |  nan           | nan          | False             |
| candidate_rille      |  nan           | nan          | False             |

### SmallUNet-Best vs Panoptic-Best

| class                |   statistic |    p_value | significant_005   |
|:---------------------|------------:|-----------:|:------------------|
| impact_crater        |      153635 |   0        | True              |
| pit_skylight         |         nan | nan        | False             |
| wrinkle_ridge        |        2582 |   0.209058 | False             |
| lobate_scarp         |         nan | nan        | False             |
| irregular_mare_patch |         nan | nan        | False             |
| apollo_site          |         nan | nan        | False             |
| candidate_rille      |         nan | nan        | False             |

### MaskRCNN-Best vs Panoptic-Best

| class                |   statistic |    p_value | significant_005   |
|:---------------------|------------:|-----------:|:------------------|
| impact_crater        |      142452 |   0        | True              |
| pit_skylight         |         nan | nan        | False             |
| wrinkle_ridge        |         982 |   0.243179 | False             |
| lobate_scarp         |         nan | nan        | False             |
| irregular_mare_patch |         nan | nan        | False             |
| apollo_site          |         nan | nan        | False             |
| candidate_rille      |         nan | nan        | False             |

