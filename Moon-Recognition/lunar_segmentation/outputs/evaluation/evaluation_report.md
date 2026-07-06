# Professional Evaluation Report: Marius Hills

- **Evaluated Tiles**: 3115
- **Decision Threshold**: 0.75
- **Bootstrap Confidence**: 0.95 (1000 resamples)

## Global & Per-Class Metrics

| model          |   (np.str_('apollo_site'), 'Dice') |   (np.str_('apollo_site'), 'F1') |   (np.str_('apollo_site'), 'IoU') |   (np.str_('apollo_site'), 'Precision') |   (np.str_('apollo_site'), 'Recall') |   (np.str_('candidate_rille'), 'Dice') |   (np.str_('candidate_rille'), 'F1') |   (np.str_('candidate_rille'), 'IoU') |   (np.str_('candidate_rille'), 'Precision') |   (np.str_('candidate_rille'), 'Recall') |   (np.str_('impact_crater'), 'Dice') |   (np.str_('impact_crater'), 'F1') |   (np.str_('impact_crater'), 'IoU') |   (np.str_('impact_crater'), 'Precision') |   (np.str_('impact_crater'), 'Recall') |   (np.str_('lobate_scarp'), 'Dice') |   (np.str_('lobate_scarp'), 'F1') |   (np.str_('lobate_scarp'), 'IoU') |   (np.str_('lobate_scarp'), 'Precision') |   (np.str_('lobate_scarp'), 'Recall') |   (np.str_('pit_skylight'), 'Dice') |   (np.str_('pit_skylight'), 'F1') |   (np.str_('pit_skylight'), 'IoU') |   (np.str_('pit_skylight'), 'Precision') |   (np.str_('pit_skylight'), 'Recall') |   (np.str_('wrinkle_ridge'), 'Dice') |   (np.str_('wrinkle_ridge'), 'F1') |   (np.str_('wrinkle_ridge'), 'IoU') |   (np.str_('wrinkle_ridge'), 'Precision') |   (np.str_('wrinkle_ridge'), 'Recall') |
|:---------------|-----------------------------------:|---------------------------------:|----------------------------------:|----------------------------------------:|-------------------------------------:|---------------------------------------:|-------------------------------------:|--------------------------------------:|--------------------------------------------:|-----------------------------------------:|-------------------------------------:|-----------------------------------:|------------------------------------:|------------------------------------------:|---------------------------------------:|------------------------------------:|----------------------------------:|-----------------------------------:|-----------------------------------------:|--------------------------------------:|------------------------------------:|----------------------------------:|-----------------------------------:|-----------------------------------------:|--------------------------------------:|-------------------------------------:|-----------------------------------:|------------------------------------:|------------------------------------------:|---------------------------------------:|
| MaskRCNN-Best  |                             0.0000 |                           0.0000 |                            0.0000 |                                  1.0000 |                               0.0000 |                               nan      |                             nan      |                              nan      |                                    nan      |                                 nan      |                               0.6437 |                             0.6499 |                              0.5758 |                                    0.9225 |                                 0.6137 |                              0.0000 |                            0.0000 |                             0.0000 |                                   1.0000 |                                0.0000 |                              0.0000 |                            0.0000 |                             0.0000 |                                   1.0000 |                                0.0000 |                               0.1357 |                             0.1586 |                              0.0901 |                                    0.7697 |                                 0.1864 |
| Panoptic-Best  |                             0.0000 |                           0.0000 |                            0.0000 |                                  1.0000 |                               0.0000 |                               nan      |                             nan      |                              nan      |                                    nan      |                                 nan      |                               0.9275 |                             0.9275 |                              0.9032 |                                    0.9032 |                                 1.0000 |                              0.0000 |                            0.0000 |                             0.0000 |                                   1.0000 |                                0.0000 |                              0.0006 |                            0.0006 |                             0.0003 |                                   0.2695 |                                0.7258 |                               0.0022 |                             0.0221 |                              0.0012 |                                    0.9566 |                                 0.0050 |
| SmallUNet-Best |                             0.0000 |                           0.0000 |                            0.0000 |                                  1.0000 |                               0.0000 |                                 0.0000 |                               0.0000 |                                0.0000 |                                      0.0000 |                                   1.0000 |                               0.9097 |                             0.9097 |                              0.8707 |                                    0.9027 |                                 0.9593 |                              0.0000 |                            0.0000 |                             0.0000 |                                   1.0000 |                                0.0000 |                              0.0001 |                            0.0077 |                             0.0000 |                                   0.0000 |                                0.9921 |                               0.0000 |                             0.0000 |                              0.0000 |                                    1.0000 |                                 0.0000 |

## Pairwise Statistical Significance (Wilcoxon Signed-Rank Test)

Tests the null hypothesis that the metric distribution medians are equal. Significant differences ($p < 0.05$) are highlighted.

### SmallUNet-Best vs MaskRCNN-Best

| class                |   statistic |       p_value | significant_005   |
|:---------------------|------------:|--------------:|:------------------|
| impact_crater        |       77142 |   0           | True              |
| pit_skylight         |          19 |   0.431641    | False             |
| wrinkle_ridge        |          15 |   6.36085e-15 | True              |
| lobate_scarp         |         nan | nan           | False             |
| irregular_mare_patch |         nan | nan           | False             |
| apollo_site          |         nan | nan           | False             |
| candidate_rille      |         nan | nan           | False             |

### SmallUNet-Best vs Panoptic-Best

| class                |   statistic |    p_value | significant_005   |
|:---------------------|------------:|-----------:|:------------------|
| impact_crater        |         586 |   0        | True              |
| pit_skylight         |          99 |   0.840822 | False             |
| wrinkle_ridge        |         nan | nan        | False             |
| lobate_scarp         |         nan | nan        | False             |
| irregular_mare_patch |         nan | nan        | False             |
| apollo_site          |         nan | nan        | False             |
| candidate_rille      |         nan | nan        | False             |

### MaskRCNN-Best vs Panoptic-Best

| class                |   statistic |       p_value | significant_005   |
|:---------------------|------------:|--------------:|:------------------|
| impact_crater        |        4090 |   0           | True              |
| pit_skylight         |         nan | nan           | False             |
| wrinkle_ridge        |          15 |   6.36085e-15 | True              |
| lobate_scarp         |         nan | nan           | False             |
| irregular_mare_patch |         nan | nan           | False             |
| apollo_site          |         nan | nan           | False             |
| candidate_rille      |         nan | nan           | False             |

