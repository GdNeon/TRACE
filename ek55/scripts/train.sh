python main_alltime_alt_ens_add_late_fusion.py train /disk/gao1/Transformer/data/ek55/ ../output/abtest/stage1 --lr_scheduler --weight_mse_by_time 0
python main_stage2_weightsum_add_late_fusion.py train /disk/gao1/Transformer/data/ek55/ ../output/abtest/stage1 stage2 --load_fusion_weight --lr_scheduler