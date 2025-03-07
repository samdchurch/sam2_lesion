import os
import json
import numpy as np
import matplotlib.pyplot as plt

['deep lesion', 'bone', 'pancreas', 'kidney', 'lung', 'liver', 'colon', 'ABD', 'MED']
#LOG_DIR = '/app/UserData/Sam/sam2_resources/logs'
LOG_DIR = 'B:/Sam/sam2_resources/logs'
def boxplot_results(run_name):
    results_dir = os.path.join(LOG_DIR, run_name, 'results')
    dice_3d_file = os.path.join(results_dir, 'dice_3d.json')
    with open(dice_3d_file) as f:
        dice_3d_all = json.load(f)
    scores = [float(v) for v in dice_3d_all.values()]

    all_scores = {'deep lesion': [], 'bone': [], 'pancreas': [], 'kidney': [], 'lung': [], 'liver': [], 'colon': [], 'ABD': [], 'MED': []}
    for filename, score in dice_3d_all.items():
        if 'kidney' in filename:
            all_scores['kidney'].append(score)
        elif 'bone' in filename:
            all_scores['bone'].append(score)
        elif 'pancreas' in filename:
            all_scores['pancreas'].append(score)
        elif 'lidcidri' in filename or 'lung' in filename:
            all_scores['lung'].append(score)
        elif 'liver' in filename:
            all_scores['liver'].append(score)
        elif 'colon' in filename:
            all_scores['colon'].append(score)
        elif 'ABD' in filename:
            all_scores['ABD'].append(score)
        elif 'MED' in filename:
            all_scores['MED'].append(score)
        else:
            all_scores['deep lesion'].append(score)
    
    scores = [score_list for cat, score_list in all_scores.items()]
    cat = [f'{cat}: {np.mean(vals):.3f}' for cat, vals in all_scores.items()]
    plt.boxplot(scores, labels=cat, vert=False)
    plt.tight_layout()
    plt.savefig('example.png')
    # print(np.mean([float(v) for v in dice_3d_all.values()]))
    # print(np.std([float(v) for v in dice_3d_all.values()]))
    # plt.boxplot(scores, vert=False)
    # plt.savefig('example.png')

if __name__ == '__main__':
    run_name = 'size-tiny_subset-all_frames-12_baselr-5e-06_visionlr-3e-06_anno-line_affine-50-20_multi-False_lora-False-8_num-ortho-0'
    boxplot_results(run_name)