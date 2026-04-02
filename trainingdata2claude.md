Devices: [CudaDevice(id=0), CudaDevice(id=1), CudaDevice(id=2), CudaDevice(id=3)]
Loaded 12043 records from data/training/expert_games.jsonl
Train records: 10839
Val records:   1204
Training mode: pmap across 4 local device(s)
Epoch 001 [Phase 1] policy_loss=2.5629 value_loss=1.4194 top1_acc=0.0106 | val_policy_loss=2.0305 val_value_loss=1.2894 val_top1_acc=0.0196
Epoch 002 [Phase 1] policy_loss=1.8957 value_loss=1.3071 top1_acc=0.0074 | val_policy_loss=1.8397 val_value_loss=1.1914 val_top1_acc=0.0083
Epoch 003 [Phase 1] policy_loss=1.8045 value_loss=1.3097 top1_acc=0.0100 | val_policy_loss=1.8163 val_value_loss=1.2706 val_top1_acc=0.0286
Epoch 004 [Phase 1] policy_loss=1.7912 value_loss=1.3014 top1_acc=0.0044 | val_policy_loss=1.8219 val_value_loss=1.1784 val_top1_acc=0.0000
Epoch 005 [Phase 1] policy_loss=1.7820 value_loss=1.2844 top1_acc=0.0026 | val_policy_loss=1.8146 val_value_loss=1.1956 val_top1_acc=0.0000
Epoch 006 [Phase 1] policy_loss=1.7744 value_loss=1.2491 top1_acc=0.0064 | val_policy_loss=1.8464 val_value_loss=1.1658 val_top1_acc=0.0007
Epoch 007 [Phase 1] policy_loss=1.7811 value_loss=1.3244 top1_acc=0.0024 | val_policy_loss=1.8389 val_value_loss=1.1742 val_top1_acc=0.0013
Epoch 008 [Phase 1] policy_loss=1.7781 value_loss=1.2874 top1_acc=0.0038 | val_policy_loss=1.8121 val_value_loss=1.1698 val_top1_acc=0.0020
Epoch 009 [Phase 1] policy_loss=1.7746 value_loss=1.2693 top1_acc=0.0048 | val_policy_loss=1.8017 val_value_loss=1.2460 val_top1_acc=0.0000
Epoch 010 [Phase 1] policy_loss=1.7733 value_loss=1.3216 top1_acc=0.0019 | val_policy_loss=1.8025 val_value_loss=1.1999 val_top1_acc=0.0026
Saved checkpoint: checkpoints/bc/epoch_010.pkl
Epoch 011 [Phase 2] policy_loss=1.7771 value_loss=1.1552 top1_acc=0.0014 | val_policy_loss=1.8284 val_value_loss=0.9772 val_top1_acc=0.0013
Epoch 012 [Phase 2] policy_loss=1.7729 value_loss=0.9911 top1_acc=0.0017 | val_policy_loss=1.8382 val_value_loss=0.8163 val_top1_acc=0.0248
Epoch 013 [Phase 2] policy_loss=1.7690 value_loss=0.7999 top1_acc=0.0040 | val_policy_loss=1.8115 val_value_loss=0.7443 val_top1_acc=0.0020
Epoch 014 [Phase 2] policy_loss=1.7689 value_loss=0.7648 top1_acc=0.0022 | val_policy_loss=1.8177 val_value_loss=0.7370 val_top1_acc=0.0095
Epoch 015 [Phase 2] policy_loss=1.7659 value_loss=0.7660 top1_acc=0.0041 | val_policy_loss=1.8341 val_value_loss=0.7575 val_top1_acc=0.0020
Epoch 016 [Phase 2] policy_loss=1.7644 value_loss=0.7731 top1_acc=0.0026 | val_policy_loss=1.8057 val_value_loss=0.7169 val_top1_acc=0.0000
Epoch 017 [Phase 2] policy_loss=1.7596 value_loss=0.7998 top1_acc=0.0022 | val_policy_loss=1.8345 val_value_loss=0.7766 val_top1_acc=0.0050
Epoch 018 [Phase 2] policy_loss=1.7722 value_loss=0.7730 top1_acc=0.0064 | val_policy_loss=1.8066 val_value_loss=0.7607 val_top1_acc=0.0007
Epoch 019 [Phase 2] policy_loss=1.7609 value_loss=0.7680 top1_acc=0.0012 | val_policy_loss=1.8141 val_value_loss=0.8052 val_top1_acc=0.0013
Epoch 020 [Phase 2] policy_loss=1.7578 value_loss=0.7629 top1_acc=0.0023 | val_policy_loss=1.8127 val_value_loss=0.7396 val_top1_acc=0.0050
Saved checkpoint: checkpoints/bc/epoch_020.pkl
Epoch 021 [Phase 2] policy_loss=1.7650 value_loss=0.7465 top1_acc=0.0017 | val_policy_loss=1.8234 val_value_loss=0.7049 val_top1_acc=0.0013
Epoch 022 [Phase 2] policy_loss=1.7609 value_loss=0.7425 top1_acc=0.0030 | val_policy_loss=1.8114 val_value_loss=0.7803 val_top1_acc=0.0050
Epoch 023 [Phase 2] policy_loss=1.7579 value_loss=0.7532 top1_acc=0.0025 | val_policy_loss=1.8143 val_value_loss=0.7009 val_top1_acc=0.0007
Epoch 024 [Phase 2] policy_loss=1.7582 value_loss=0.7524 top1_acc=0.0020 | val_policy_loss=1.8159 val_value_loss=1.0284 val_top1_acc=0.0007
Epoch 025 [Phase 2] policy_loss=1.7599 value_loss=0.7948 top1_acc=0.0030 | val_policy_loss=1.8331 val_value_loss=0.7362 val_top1_acc=0.0000
Epoch 026 [Phase 2] policy_loss=1.7534 value_loss=0.7393 top1_acc=0.0017 | val_policy_loss=1.8101 val_value_loss=0.7244 val_top1_acc=0.0007
Epoch 027 [Phase 2] policy_loss=1.7557 value_loss=0.7514 top1_acc=0.0024 | val_policy_loss=1.8145 val_value_loss=0.7133 val_top1_acc=0.0050
Epoch 028 [Phase 2] policy_loss=1.7517 value_loss=0.7535 top1_acc=0.0022 | val_policy_loss=1.8131 val_value_loss=0.7119 val_top1_acc=0.0000
Epoch 029 [Phase 2] policy_loss=1.7520 value_loss=0.7405 top1_acc=0.0023 | val_policy_loss=1.8131 val_value_loss=0.7341 val_top1_acc=0.0007
Epoch 030 [Phase 2] policy_loss=1.7587 value_loss=0.7360 top1_acc=0.0023 | val_policy_loss=1.8199 val_value_loss=0.6951 val_top1_acc=0.0044
Saved checkpoint: checkpoints/bc/epoch_030.pkl
Epoch 031 [Phase 3] policy_loss=1.8636 value_loss=1.3034 top1_acc=0.0030 | val_policy_loss=1.9373 val_value_loss=1.1913 val_top1_acc=0.0286
Epoch 032 [Phase 3] policy_loss=1.8019 value_loss=1.3375 top1_acc=0.0325 | val_policy_loss=1.8169 val_value_loss=1.1933 val_top1_acc=0.0324
Epoch 033 [Phase 3] policy_loss=1.7682 value_loss=1.3422 top1_acc=0.0320 | val_policy_loss=1.8374 val_value_loss=1.1933 val_top1_acc=0.0324
Epoch 034 [Phase 3] policy_loss=1.7645 value_loss=1.3524 top1_acc=0.0319 | val_policy_loss=1.8121 val_value_loss=1.1933 val_top1_acc=0.0324
Epoch 035 [Phase 3] policy_loss=1.7559 value_loss=1.3483 top1_acc=0.0323 | val_policy_loss=1.8196 val_value_loss=1.1933 val_top1_acc=0.0330
Epoch 036 [Phase 3] policy_loss=1.7497 value_loss=1.3439 top1_acc=0.0332 | val_policy_loss=1.8196 val_value_loss=1.1933 val_top1_acc=0.0330
Epoch 037 [Phase 3] policy_loss=1.7526 value_loss=1.3349 top1_acc=0.0303 | val_policy_loss=1.8178 val_value_loss=1.1819 val_top1_acc=0.0330
Epoch 038 [Phase 3] policy_loss=1.7538 value_loss=1.3082 top1_acc=0.0290 | val_policy_loss=1.8038 val_value_loss=1.1407 val_top1_acc=0.0229
Epoch 039 [Phase 3] policy_loss=1.7502 value_loss=1.3126 top1_acc=0.0264 | val_policy_loss=1.8325 val_value_loss=1.1507 val_top1_acc=0.0330
Epoch 040 [Phase 3] policy_loss=1.7534 value_loss=1.3037 top1_acc=0.0289 | val_policy_loss=1.8150 val_value_loss=1.2382 val_top1_acc=0.0186
Saved checkpoint: checkpoints/bc/epoch_040.pkl
Epoch 041 [Phase 3] policy_loss=1.7683 value_loss=1.3489 top1_acc=0.0276 | val_policy_loss=1.8163 val_value_loss=1.1933 val_top1_acc=0.0109
Epoch 042 [Phase 3] policy_loss=1.7533 value_loss=1.3486 top1_acc=0.0149 | val_policy_loss=1.8252 val_value_loss=1.1933 val_top1_acc=0.0330
Epoch 043 [Phase 3] policy_loss=1.7484 value_loss=1.3362 top1_acc=0.0186 | val_policy_loss=1.8196 val_value_loss=1.1933 val_top1_acc=0.0013
Epoch 044 [Phase 3] policy_loss=1.7474 value_loss=1.3439 top1_acc=0.0118 | val_policy_loss=1.8024 val_value_loss=1.1932 val_top1_acc=0.0253
Epoch 045 [Phase 3] policy_loss=1.7460 value_loss=1.3456 top1_acc=0.0210 | val_policy_loss=1.8099 val_value_loss=1.1896 val_top1_acc=0.0134
Epoch 046 [Phase 3] policy_loss=1.7451 value_loss=1.3291 top1_acc=0.0101 | val_policy_loss=1.8135 val_value_loss=1.1827 val_top1_acc=0.0330
Epoch 047 [Phase 3] policy_loss=1.7459 value_loss=1.3206 top1_acc=0.0232 | val_policy_loss=1.8143 val_value_loss=1.1732 val_top1_acc=0.0089
Epoch 048 [Phase 3] policy_loss=1.7458 value_loss=1.3244 top1_acc=0.0177 | val_policy_loss=1.8170 val_value_loss=1.1583 val_top1_acc=0.0330
Epoch 049 [Phase 3] policy_loss=1.7515 value_loss=1.2992 top1_acc=0.0156 | val_policy_loss=1.8130 val_value_loss=1.1717 val_top1_acc=0.0324
Epoch 050 [Phase 3] policy_loss=1.7454 value_loss=1.2747 top1_acc=0.0219 | val_policy_loss=1.8260 val_value_loss=1.1462 val_top1_acc=0.0115
Saved checkpoint: checkpoints/bc/epoch_050.pkl