# train on MIMIC-III
python train_tacco.py --dname=mimic --epochs=500 --All_num_layers 3 --cuda=1 --num_labels=25 --num_nodes=7423 --feature_dim=256 --heads=4 --warmup 100 --MLP_num_layers 2 --alpha 10 --num_cluster 5 --beta 0.1 --MLP_hidden 48

# train on CRADLE
python train_tacco.py --dname=cradle --epochs=500 --All_num_layers 3 --cuda=1 --num_labels=1 --num_nodes=12725 --feature_dim=256 --heads=4 --warmup 100 --MLP_num_layers 2 --alpha 10 --num_cluster 5 --beta 0.1 --MLP_hidden 48
