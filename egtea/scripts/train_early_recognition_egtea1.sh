#CUDA_VISIBLE_DEVICES=1 python pretrain.py train ../Transformer/data/egtea ../Transformer/pretrain_model --data_id 1
CUDA_VISIBLE_DEVICES=1 python main_alltime_alt_ens_add_late_fusion.py train ../Transformer/data/egtea ../Transformer/models/egtea --path_to_pretrain ../Transformer/pretrain_model --lr 0.00002 --data_id 1 --lr_scheduler
CUDA_VISIBLE_DEVICES=1 python main_stage2_weightsum_add_late_fusion.py train ../Transformer/data/egtea ../Transformer/models/egtea ../Transformer/models/egtea --data_id 1 --lr_scheduler --load_fusion_weight

CUDA_VISIBLE_DEVICES=1 python main_alltime_alt_ens_add_late_fusion.py train ../Transformer/data/egtea ../Transformer/models/egtea --path_to_pretrain ../Transformer/pretrain_model --lr 0.00004 --data_id 1 --lr_scheduler
CUDA_VISIBLE_DEVICES=1 python main_stage2_weightsum_add_late_fusion.py train ../Transformer/data/egtea ../Transformer/models/egtea ../Transformer/models/egtea --data_id 1 --lr_scheduler --load_fusion_weight 

CUDA_VISIBLE_DEVICES=1 python main_alltime_alt_ens_add_late_fusion.py train ../Transformer/data/egtea ../Transformer/models/egtea --path_to_pretrain ../Transformer/pretrain_model --lr 0.00006 --data_id 1 --lr_scheduler
CUDA_VISIBLE_DEVICES=1 python main_stage2_weightsum_add_late_fusion.py train ../Transformer/data/egtea ../Transformer/models/egtea ../Transformer/models/egtea --data_id 1 --lr_scheduler --load_fusion_weight 

CUDA_VISIBLE_DEVICES=1 python main_alltime_alt_ens_add_late_fusion.py train ../Transformer/data/egtea ../Transformer/models/egtea --path_to_pretrain ../Transformer/pretrain_model --lr 0.00008 --data_id 1 --lr_scheduler
CUDA_VISIBLE_DEVICES=1 python main_stage2_weightsum_add_late_fusion.py train ../Transformer/data/egtea ../Transformer/models/egtea ../Transformer/models/egtea --data_id 1 --lr_scheduler --load_fusion_weight 

CUDA_VISIBLE_DEVICES=1 python main_alltime_alt_ens_add_late_fusion.py train ../Transformer/data/egtea ../Transformer/models/egtea --path_to_pretrain ../Transformer/pretrain_model --lr 0.0001 --data_id 1 --lr_scheduler
CUDA_VISIBLE_DEVICES=1 python main_stage2_weightsum_add_late_fusion.py train ../Transformer/data/egtea ../Transformer/models/egtea ../Transformer/models/egtea --data_id 1 --lr_scheduler --load_fusion_weight