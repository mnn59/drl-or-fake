# running PPO
#python3 main.py --use-gae --num-mini-batch 4 --use-linear-lr-decay --num-env-steps 100000 --env-name Abi --log-dir ./log/test --demand-matrix Abi_500.txt --model-save-path ./model/test 

 
# 10000000


##############
# Fig.5 a,d down safe
# python3 main.py \
#     --env-name Abi \
#     --demand-matrix Abi_500.txt \
#     --num-env-steps 300000 \
#     --num-steps 512 \
#     --num-mini-batch 4 \
#     --num-pretrain-epochs 30 \
#     --num-pretrain-steps 128 \
#     --lr 2.5e-5 \
#     --use-gae \
#     --use-linear-lr-decay \
#     --log-dir ./log/initialization \
#     --model-save-path ./model/initialization \
#     --seed 1


# Fig.5 b,e down safe
# python3 main.py \
#     --env-name Abi \
#     --demand-matrix Abi_500.txt \
#     --num-env-steps 180000 \
#     --use-gae \
#     --log-dir ./log/link_failure \
#     --model-load-path ./model/initialization \
#     --model-save-path ./model/link_failure \



# Fig.5 c,f down safe
python3 main.py \
    --env-name Abi \
    --demand-matrix Abi_500.txt \
    --num-env-steps 180000 \
    --use-gae \
    --log-dir ./log/traffic_change \
    --model-load-path ./model/initialization \
    --model-save-path ./model/traffic_change \
