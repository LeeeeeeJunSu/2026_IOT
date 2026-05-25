# Geometry-aware GT0 baseline analysis

GT0 is treated as empty-room baseline. Values below are GT - GT0 deltas unless noted.

## Top ESP changes by GT

### GT 1 (front-left)
| rank | node | feature_L2_delta | amp_mean_delta | rssi_delta_db | group |
|---:|---:|---:|---:|---:|---|
| 1 | 8 | 2.719 | +1.387 | -0.113 | left wall ESP5-8 |
| 2 | 5 | 2.238 | +0.052 | -0.188 | left wall ESP5-8 |
| 3 | 7 | 2.148 | +0.908 | +0.810 | left wall ESP5-8 |
| 4 | 4 | 1.866 | -1.007 | +0.594 | right wall ESP1-4 |
| 5 | 6 | 1.637 | -0.198 | -0.855 | left wall ESP5-8 |

### GT 2 (front-right)
| rank | node | feature_L2_delta | amp_mean_delta | rssi_delta_db | group |
|---:|---:|---:|---:|---:|---|
| 1 | 8 | 3.964 | -0.953 | +0.208 | left wall ESP5-8 |
| 2 | 9 | 1.954 | -0.234 | +2.275 | front ESP9 |
| 3 | 1 | 1.550 | +0.767 | +0.866 | right wall ESP1-4 |
| 4 | 5 | 1.498 | +0.680 | -0.319 | left wall ESP5-8 |
| 5 | 3 | 1.405 | +0.702 | -0.406 | right wall ESP1-4 |

### GT 3 (middle-left)
| rank | node | feature_L2_delta | amp_mean_delta | rssi_delta_db | group |
|---:|---:|---:|---:|---:|---|
| 1 | 2 | 5.413 | +3.127 | +2.428 | right wall ESP1-4 |
| 2 | 5 | 3.880 | +1.932 | -0.776 | left wall ESP5-8 |
| 3 | 8 | 3.416 | -0.475 | -0.491 | left wall ESP5-8 |
| 4 | 4 | 2.291 | +1.273 | +1.343 | right wall ESP1-4 |
| 5 | 9 | 2.160 | -0.260 | +2.688 | front ESP9 |

### GT 4 (middle-right)
| rank | node | feature_L2_delta | amp_mean_delta | rssi_delta_db | group |
|---:|---:|---:|---:|---:|---|
| 1 | 2 | 3.366 | +1.690 | +2.273 | right wall ESP1-4 |
| 2 | 5 | 2.777 | +1.061 | -1.220 | left wall ESP5-8 |
| 3 | 9 | 2.674 | -0.123 | +2.769 | front ESP9 |
| 4 | 4 | 2.617 | +0.800 | +0.594 | right wall ESP1-4 |
| 5 | 8 | 1.672 | -0.289 | -1.563 | left wall ESP5-8 |

### GT 5 (back-left)
| rank | node | feature_L2_delta | amp_mean_delta | rssi_delta_db | group |
|---:|---:|---:|---:|---:|---|
| 1 | 2 | 4.344 | +2.374 | +2.818 | right wall ESP1-4 |
| 2 | 9 | 3.639 | -0.658 | +1.080 | front ESP9 |
| 3 | 8 | 3.590 | -0.862 | -0.343 | left wall ESP5-8 |
| 4 | 6 | 2.599 | -0.163 | -1.843 | left wall ESP5-8 |
| 5 | 4 | 2.061 | +0.894 | +0.251 | right wall ESP1-4 |

### GT 6 (back-right)
| rank | node | feature_L2_delta | amp_mean_delta | rssi_delta_db | group |
|---:|---:|---:|---:|---:|---|
| 1 | 8 | 3.492 | -0.963 | -0.348 | left wall ESP5-8 |
| 2 | 9 | 2.955 | -0.409 | +2.698 | front ESP9 |
| 3 | 2 | 2.821 | +1.341 | -0.463 | right wall ESP1-4 |
| 4 | 5 | 2.678 | +1.250 | -1.270 | left wall ESP5-8 |
| 5 | 4 | 2.216 | -1.074 | -1.221 | right wall ESP1-4 |

## Group averages

| GT | pos | right wall L2 | left wall L2 | front ESP9 L2 | back ESP10 L2 | right amp d | left amp d | ESP9 amp d | ESP10 amp d |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | front-left | 0.957 | 2.186 | 1.403 | 0.986 | -0.114 | +0.537 | +0.781 | +0.171 |
| 2 | front-right | 1.277 | 1.915 | 1.954 | 1.234 | +0.705 | -0.075 | -0.234 | +0.697 |
| 3 | middle-left | 2.305 | 2.658 | 2.160 | 0.996 | +1.191 | +0.257 | -0.260 | +0.541 |
| 4 | middle-right | 2.031 | 1.711 | 2.674 | 1.528 | +0.654 | +0.181 | -0.123 | +0.746 |
| 5 | back-left | 2.472 | 2.390 | 3.639 | 1.194 | +0.630 | -0.300 | -0.658 | +0.226 |
| 6 | back-right | 1.719 | 2.226 | 2.955 | 1.284 | -0.046 | -0.040 | -0.409 | -0.343 |

## Specific checks

- Front cells GT1/GT2 on ESP9: GT1 L2=1.403, amp=+0.781, RSSI=+2.254 dB; GT2 L2=1.954, amp=-0.234, RSSI=+2.275 dB
- Left cells GT1/GT3/GT5 on ESP5-8: GT1 left-wall avg L2=2.186, avg amp=+0.537, strongest ESP8 L2=2.719; GT3 left-wall avg L2=2.658, avg amp=+0.257, strongest ESP5 L2=3.880; GT5 left-wall avg L2=2.390, avg amp=-0.300, strongest ESP8 L2=3.590
- Right cells GT2/GT4/GT6 on ESP1-4: GT2 right-wall avg L2=1.277, avg amp=+0.705, strongest ESP1 L2=1.550; GT4 right-wall avg L2=2.031, avg amp=+0.654, strongest ESP2 L2=3.366; GT6 right-wall avg L2=1.719, avg amp=-0.046, strongest ESP2 L2=2.821