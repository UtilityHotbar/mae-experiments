import os
import ast
import pickle
import numpy as np
import re
import pprint

LOG_DIR = 'logs'
OUTPUT_DIR = 'output'
MODEL_NAME = 'vit_base_patch16'
IMAGE_DIR = 'imagenet_limited/val'

final_report = []

def report_print(*args):
    print(*args)
    final_report.append(' '.join([str(arg) for arg in args]))

results = {}
acc1 = {}
acc5 = {}
acc1_lowest = [-1, 101]
acc1_highest = [-1, 0]
acc5_lowest = [-1, 101]
acc5_highest = [-1, 0]

image_target_classes = {}
images_with_results = {}
highest_delta_shift = ['', 0, 0]

print('Finding test images.')
curr_class = 0
for thing in os.scandir(IMAGE_DIR):
    if thing.is_dir():
        for subthing in os.scandir(thing.path):
            if subthing.is_file() and subthing.name.endswith('.JPEG'):
                image_target_classes[subthing.path] = curr_class
        curr_class += 1

image_names = list(image_target_classes.keys())
report_print(len(image_names))

print('Aggregating results over runs.')
for thing in os.scandir(LOG_DIR):
    if thing.is_file() and thing.name.startswith(MODEL_NAME):
        print('LOG', thing.name)
        patch_id = thing.name[len(MODEL_NAME):].split('_')[1]
        with open(thing.path) as f:
            try:
                patch_id = int(patch_id)
                results[patch_id] = ast.literal_eval(f.read())
                acc1[patch_id] = results[patch_id]['acc1']
                acc5[patch_id] = results[patch_id]['acc5']
                if acc1[patch_id] > acc1_highest[1]:
                    acc1_highest = [patch_id, acc1[patch_id]]
                if acc1[patch_id] < acc1_lowest[1]:
                    acc1_lowest = [patch_id, acc1[patch_id]]
                if acc5[patch_id] > acc5_highest[1]:
                    acc5_highest = [patch_id, acc1[patch_id]]
                if acc5[patch_id] < acc5_lowest[1]:
                    acc5_lowest = [patch_id, acc1[patch_id]]
            except ValueError:
                results[patch_id] = ast.literal_eval(f.read())  # this means the patch_id is "eval" and should not be considered as part of the experiment results
            
        FLNM = ''
        for filename in os.listdir(OUTPUT_DIR):
            if re.match(f"{MODEL_NAME}_{patch_id}_\S+", filename):
                FLNM = filename
        if not FLNM:
            raise RuntimeError
        else:
            print('OUTPUT', FLNM)
        with open(OUTPUT_DIR + '/' + FLNM, 'rb') as g:
            curr_img = 0
            for result_group in pickle.load(g):
                for elem in result_group:
                    # print(elem.shape)
                    l_elem = np.array(elem)
                    # print(l_elem[:10])
                    # print(type(l_elem))
                    # print(len(l_elem), max(l_elem), )
                    # print(l_elem.index(max(l_elem)))
                    # input()
                    # l_elem = np.array(elem)
                    im_name = image_names[curr_img]
                    lmax = np.argmax(l_elem)
                    lmin = np.argmin(l_elem)
                    try:
                        _ = images_with_results[im_name]
                    except:
                        images_with_results[im_name] = {}
                    images_with_results[im_name][patch_id] = {'max_id': lmax, 'max': float(l_elem[lmax]), 'min_id': min, 'min': float(l_elem[lmin])}
                    curr_img += 1
            print(curr_img)

print('Calculating confidence deltas.')
# Calculate confidence deltas
shifts = 0
non_shifts = 0
total_delta_decrease = 0
patch_shifts = {}
patch_confidence_decreases = {}
images_with_deltas = {}

for image_result in images_with_results:
    images_with_deltas[image_result] = {}
    for patch in images_with_results[image_result]:
        # print(images_with_results[image_result]['eval'], images_with_results[image_result][patch])
        # print(images_with_results[image_result]['eval']['max_id'], images_with_results[image_result][patch]['max_id'],images_with_results[image_result]['eval']['max_id']==images_with_results[image_result][patch]['max_id'])

        # input()
        if images_with_results[image_result]['eval']['max_id'] == images_with_results[image_result][patch]['max_id']:
            delta_shift = images_with_results[image_result]['eval']['max'] - images_with_results[image_result][patch]['max']
            images_with_deltas[image_result][patch] = delta_shift
            if delta_shift > highest_delta_shift[2]:
                highest_delta_shift = [image_result, patch, delta_shift]
            total_delta_decrease -= delta_shift
            non_shifts += 1
            try:
                patch_confidence_decreases[patch]['total'] += delta_shift
                patch_confidence_decreases[patch]['n'] += 1
            except:
                patch_confidence_decreases[patch] = {'total': delta_shift, 'n': 1}

        else:
            # report_print(f"Cutting patch {patch} for image {image_result} caused a shift in prediction from class {images_with_results[image_result]['eval']['max_id']} with confidence {images_with_results[image_result]['eval']['max']} to class {images_with_results[image_result][patch]['max_id']} with confidence {images_with_results[image_result][patch]['max']}")
            images_with_deltas[image_result][patch] = "SHIFTED"
            shifts += 1
            try:
                patch_shifts[patch] += 1
            except KeyError:
                patch_shifts[patch] = 1

avg_delta_decrease = total_delta_decrease / non_shifts

patches = list(results.keys())
report_print('PATCHES CUT:', ', '.join([str(_) for _ in patches]))
report_print('===')
report_print('OVERALL STATISTICS:')
report_print('Baseline acc1:', results['eval']['acc1'],'%')
report_print('Highest acc1: Cut patch', acc1_highest[0], 'with accuracy', acc1_highest[1], '%')
report_print('Lowest acc1: Cut patch', acc1_lowest[0], 'with accuracy', acc1_lowest[1], '%')
report_print('Baseline acc5:', results['eval']['acc5'],'%')
report_print('Highest acc5: Cut patch', acc5_highest[0], 'with accuracy', acc5_highest[1], '%')
report_print('Lowest acc5: Cut patch', acc5_lowest[0], 'with accuracy', acc5_lowest[1], '%')
report_print('Total cases of changed prediction due to cutting any patch: ', shifts, '/', non_shifts, '(', shifts/(shifts+non_shifts)*100, '%)')
report_print('Highest confidence shift without changing prediction:', highest_delta_shift[0], 'after cutting patch', highest_delta_shift[1], ':', highest_delta_shift[2])
report_print('Average confidence shift without changing prediction:', avg_delta_decrease)

highest = [-1, 0]
ties = []
for patch in patch_shifts:
    if patch_shifts[patch] > highest[1]:
        highest = [patch, patch_shifts[patch]]
        ties = [patch]
    elif patch_shifts[patch] == highest[1]:
        ties.append(patch)

pprint.pprint(patch_confidence_decreases)
if highest[1] > 0:
    if ties:
        report_print('Patches that caused maximum shifts in prediction:', ', '.join([f'{tie} [{patch_confidence_decreases[tie]["total"]/patch_confidence_decreases[tie]["n"]}]' for tie in ties]), f'({highest[1]} shifts)')
    else:
        report_print('Patch that caused maximum shifts in prediction:', f'{highest[0]} [{patch_confidence_decreases[highest[0]]["total"]/patch_confidence_decreases[highest[0]]["n"]}]', f'({highest[1]} shifts)')
else:
    report_print('No prediction shifts were caused by cutting out patches.')

print('Writing report...')
with open(LOG_DIR+'/'+'SUMMARY'+ '_' + MODEL_NAME + '_' + str(len(patches)), 'w') as f:
    f.write('\n'.join(final_report))
print('Done.')
