# Corrected UCI-HAR source-grouped reproduction

The official UCI test split was not opened. Values are participant-grouped
out-of-fold development evidence on the released training split.

| Model | Mean participant macro-F1 | Worst | Lower decile | Balanced accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy_joint_bilstm256_cnn128 | 0.9118 | 0.7290 | 0.8631 | 0.9148 | 0.2794 | 0.1437 | 0.0194 |
| legacy_cnn1d_h128 | 0.8852 | 0.6328 | 0.8439 | 0.8893 | 0.3828 | 0.1955 | 0.0628 |
| legacy_bilstm_h192 | 0.7084 | 0.5243 | 0.6495 | 0.7186 | 0.7031 | 0.3653 | 0.0297 |
