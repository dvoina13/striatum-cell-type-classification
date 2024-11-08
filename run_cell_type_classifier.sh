#!/bin/bash

#  run_cell_type_classifier.sh
#  
#
#  Created by Doris V on 5/22/24.
#

#source activate pytorch

seed_arr=($(seq 1 1 20))
batch_arr=(1 5 10 20)

#for seed in "${seed_arr[@]}"
#do
#    python GNN_classify.py --seed=$seed

#done

for s in "${seed_arr[@]}"
do
    echo $s
    #python GNN_classify_test.py --seed=$seed
    #python main.py --seed=$s
    python main_sparsityLayer.py --seed=$s
done