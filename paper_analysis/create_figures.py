import pandas as pd
import os
import seaborn as sns
import pickle
import torch
import torchvision
from torchvision import transforms as transforms
import matplotlib.pyplot as plt
import sklearn.metrics
import numpy as np

def load_cifar10_test_data():
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    return(torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=train_transform))

def load_black_box_output():
    original_output = pd.read_csv(os.path.join('..', 'project_data', 'black_box_output_cifar10_test.txt'), sep = '\t')
    adversarial_output_cw = pd.read_csv(os.path.join('..', 'project_data', 'black_box_output_adversarial_cw_threshold_None.txt'), sep = '\t')
    adversarial_output_lbfgs = pd.read_csv(os.path.join('..', 'project_data', 'black_box_output_adversarial_lbfgs_threshold_1.txt'), sep = '\t')

    return(original_output, adversarial_output_cw, adversarial_output_lbfgs)

def load_adversarial_data(filepath):
    # load data
    with open(filepath, 'rb') as f:
        images, labels, distances = pickle.load(f)
    return(images, labels, distances)

def plot_distance_distribution(distances, output_file):
    if np.max(distances) > 2:
        distances = [min(elem, 2) for elem in distances]
        bin_def = list(np.linspace(0,2,41))
        xtick_def = [str(elem) for elem in bin_def[:-1]] + [r'$\geq 2$']
        xtick_def = [elem if elem in ['0.0', '0.5', '1.0', '1.5', r'$\geq 2$'] else '' for elem in xtick_def]
        n, bins, patches = plt.hist(x = distances, bins = bin_def, density = 1, edgecolor = 'black', facecolor = 'blue', alpha = 0.5)
        plt.xticks(bin_def, xtick_def)
    else:
        n, bins, patches = plt.hist(x = distances, bins ='auto', density = 1, edgecolor = 'black', facecolor = 'blue', alpha = 0.5)
    plt.title(r'$\ell_2$ distance between original and adversarial images')
    plt.xlabel(r'$\ell_2$ distance')
    plt.ylabel('Density')
    plt.savefig(output_file)
    plt.close()
    return None

def generate_black_box_confusion_matrix(original_bb_predictions, adversarial_bb_predictions, output_file):
    confusion_matrix = sklearn.metrics.confusion_matrix(original_bb_predictions, adversarial_bb_predictions)
    confusion_matrix_df = pd.DataFrame(confusion_matrix,
                                       index = ['abstain', 'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
                                       columns = ['abstain', 'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    confusion_matrix_df.to_latex(output_file)
    return None

def print_original_images(cifar10, output_dir):
    for i in range(10):
        orig_image = cifar10[i][0]
        torchvision.utils.save_image(orig_image, os.path.join(output_dir, 'orig_{0}.png'.format(i)))
    return None

def print_adversarial_images(adv_images, adv_version, output_dir):
    for i in range(10):
        adv_image = torch.FloatTensor(adv_images[i])
        torchvision.utils.save_image(adv_image, os.path.join(output_dir, 'adv_{0}_{1}.png'.format(i, adv_version)))
    return None

if __name__ == '__main__':
    ''''''
    '''
    define paths
    '''
    # define output directory
    output_dir = os.path.join('..', 'figures')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    '''
    load data
    '''
    # load cifar10 data
    cifar10 = load_cifar10_test_data()

    # load adversarial data
    cw_filepath = os.path.join('..', 'substitute_model_testing', 'l2_adversarial_output_cw_threshold_None.pkl')
    lbfgs_filepath = os.path.join('..', 'substitute_model_testing', 'l2_adversarial_output_lbfgs_threshold_1.0.pkl')
    adv_images_cw, white_box_labels_cw, adv_distances_cw = load_adversarial_data(cw_filepath)
    adv_images_lbfgs, white_box_labels_lbfgs, adv_distances_lbfgs = load_adversarial_data(lbfgs_filepath)

    # load black box output
    original_bb_output, adversarial_bb_cw_output, adversarial_bb_lbfgs_output = load_black_box_output()
    original_bb_predictions = original_bb_output['predict']
    adversarial_bb_cw_predictions = adversarial_bb_cw_output['predict']
    adversarial_bb_lbfgs_predictions = adversarial_bb_lbfgs_output['predict']

    # update adversarial_bb_lbfgs_predictions/correct for elements with l2 distance > 1
    adversarial_bb_lbfgs_predictions_updated = pd.Series([adversarial_bb_lbfgs_predictions[i] if adv_distances_lbfgs[i] <= 1 else original_bb_predictions[i] for i in range(len(adversarial_bb_lbfgs_predictions))])
    adversarial_bb_lbfgs_correct = [adversarial_bb_lbfgs_output['correct'][i] if adv_distances_lbfgs[i] <= 1 else original_bb_output['correct'][i] for i in range(len(adversarial_bb_lbfgs_predictions))]


    '''
    print performance metrics
    '''
    # get proportion correct and abstention rate
    original_bb_counts = original_bb_output['predict'].value_counts()
    adversarial_bb_cw_counts = adversarial_bb_cw_output['predict'].value_counts()
    adversarial_bb_lbfgs_counts = adversarial_bb_lbfgs_predictions_updated.value_counts()

    print('Original: {0} correct, {1} abstain'.format(np.mean(original_bb_output['correct']), original_bb_counts.to_dict()[-1] / np.sum(original_bb_counts)))
    print('CW: {0} correct, {1} abstain'.format(np.mean(adversarial_bb_cw_output['correct']), adversarial_bb_cw_counts.to_dict()[-1] / np.sum(adversarial_bb_cw_counts)))
    print('TV: {0} correct, {1} abstain'.format(np.mean(adversarial_bb_lbfgs_correct), adversarial_bb_lbfgs_counts.to_dict()[-1] / np.sum(adversarial_bb_lbfgs_counts)))



    '''
    generate figures
    '''
    # plot distribution of distances from original to adversarial images
    plot_distance_distribution(distances = adv_distances_cw, output_file = os.path.join(output_dir, 'distance_distribution_cw.png'))
    plot_distance_distribution(distances = adv_distances_lbfgs, output_file = os.path.join(output_dir, 'distance_distribution_lbfgs.png'))

    # generate confusion matrix of original black box predictions and black box predictions on adversarial images
    generate_black_box_confusion_matrix(original_bb_predictions, adversarial_bb_cw_predictions, output_file = os.path.join(output_dir, 'black_box_cw_confusion_matrix.txt'))
    generate_black_box_confusion_matrix(original_bb_predictions, adversarial_bb_lbfgs_predictions_updated, output_file = os.path.join(output_dir, 'black_box_lbfgs_confusion_matrix.txt'))

    # print subset of original and adversarial images
    print_original_images(cifar10, output_dir = output_dir)
    print_adversarial_images(adv_images_cw, adv_version = 'cw', output_dir = output_dir)
    print_adversarial_images(adv_images_lbfgs, adv_version = 'lbfgs', output_dir = output_dir)
