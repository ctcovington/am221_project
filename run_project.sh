#!/usr/bin/env bash

# query black box model on CIFAR10 data
print "running black box on CIFAR10 test data"
python ./smoothing/code/predict.py cifar10 ./smoothing/models/cifar10/resnet110/noise_0.50/checkpoint.pth.tar 0.50 ./project_data/black_box_output_cifar10_test.txt --split test --alpha 0.001


# build substitute model
cd substitute_model_testing
print "building substitute model"
python model.py --lr 0.01

# run l2 attack on substitute
print "attacking substitute model"
python l2_attack.py
python l2_attack.py --threshold 1
cd ..

# query black box model on adversarial examples
python ./smoothing/code/adversarial_predict.py cifar10 ./smoothing/models/cifar10/resnet110/noise_0.50/checkpoint.pth.tar 0.50 ./project_data/black_box_output_adversarial.txt --split test --alpha 0.001 --adversarial_output ./substitute_model_testing/l2_adversarial_output_cw_threshold_None.pkl
python ./smoothing/code/adversarial_predict.py cifar10 ./smoothing/models/cifar10/resnet110/noise_0.50/checkpoint.pth.tar 0.50 ./project_data/black_box_output_adversarial_lbfgs_threshold_1.txt --split test --alpha 0.001 --threshold 1 --adversarial_output ./substitute_model_testing/l2_adversarial_output_lbfgs_threshold_1.pkl
