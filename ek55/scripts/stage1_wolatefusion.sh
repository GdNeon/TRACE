python main_alltime_alt_ens_add_late_fusion.py train /disk/gao1/Transformer/data/ek55/ ../output/abtest/stage1_wolatefusion --lr_scheduler --weight_mse_by_time 0 --weight_late_fusion 0
python main_stage2_weightsum_add_late_fusion.py train /disk/gao1/Transformer/data/ek55/ ../output/abtest/stage1_wolatefusion stage2_wolatefusion --load_fusion_weight --lr_scheduler --weight_late_fusion 0
# 这个看最好的apre_ens就行