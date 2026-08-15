import numpy as np
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import os
import xml.etree.ElementTree as ET

def get_cell_counts(anno_dir):
    """
    Function that intakes a string for the annotations directory and outputs the total RBC, WBC, and Platelet counts
    respectively. Used for distribution-matched splitting of train/test/val later on.

    OUTPUT:
    xml_data: list of dicts with primary equal to the filename minus extension.
              nested dicts under that key mirror XML format.
              Ex: {'BloodImage_00315': 
                    {'RBC1': 
                            {'name': 'RBC', 'xmin': 164, 'xmax': 297, 'ymin': 261, 'ymax': 364, 'truncated': 0, 'difficult': 0}, 
                    'RBC2': {'name': 'RBC', 'xmin': 15, 'xmax': 148, 'ymin': 66, 'ymax': 169, 'truncated': 0, 'difficult': 0}, 
                                            .
                                            .
                                            .
                    WBC1': {'name': 'WBC', 'xmin': 250, 'xmax': 487, 'ymin': 343, 'ymax': 480, 'truncated': 1, 'difficult': 0}, 
                    'Platelets1': {'name': 'Platelets', 'xmin': 567, 'xmax': 619, 'ymin': 234, 'ymax': 277, 'truncated': 0, 'difficult': 0}, 
                    'counts': {'RBC': 10, 'WBC': 1, 'Platelets': 1}}}
    rbc_counts: list of RBC counts for every image
    wbc_counts: list of WBC counts for every image
    platelet_count: list of Platelet counts for every image
    """
    # Getting Annotations and Exploring Blood Cell counts. This is to find out if the dataset is balanced or not.
    files_in_anno_dir = sorted([
            f for f in os.listdir(anno_dir) if f.endswith(".xml")
        ]) #sorts files to ensure indices are always pointing to correct filenames
    
    # Parsing the XML files and converting them to JSON format for easier analysis
    xml_data = []
    for f in files_in_anno_dir:
        with open(os.path.join(anno_dir, f)) as xml_file:
            xml_data.append({f[:-4]: {}})
            tree = ET.parse(xml_file)
            root = tree.getroot()
            names = []
            rbc_count = 0
            wbc_count = 0
            platelet_count = 0
            for b in root.findall('object'):
                name = b.find('name').text
                if name.lower() == 'RBC'.lower():
                    rbc_count += 1
                    new_key = name + str(rbc_count)
                elif name.lower() == 'WBC'.lower():
                    wbc_count += 1
                    new_key = name + str(wbc_count)
                elif name.lower() == 'Platelets'.lower():
                    platelet_count += 1
                    new_key = name + str(platelet_count)
                bounding_box = b.find('bndbox')
                xmin = int(bounding_box.find('xmin').text)
                xmax = int(bounding_box.find('xmax').text)
                ymin = int(bounding_box.find('ymin').text)
                ymax = int(bounding_box.find('ymax').text)
                trunc = int(b.find('truncated').text)
                difficult = int(b.find('difficult').text)
                names.append(name)
                xml_data[-1][f[:-4]][new_key] = {
                    'name': name,
                    'xmin': xmin,
                    'xmax': xmax,
                    'ymin': ymin,
                    'ymax': ymax,
                    'truncated': trunc,
                    'difficult': difficult
                }
            xml_data[-1][f[:-4]]['counts'] = {
                'RBC': rbc_count,
                'WBC': wbc_count,
                'Platelets': platelet_count
            }

    #Extracting the counts of each blood cell type from the JSON data
    rbc_counts = []
    wbc_counts = []
    platelet_counts = []
    for data in xml_data:
        for key in data.keys():
            rbc_counts.append(data[key]['counts']['RBC'])
            wbc_counts.append(data[key]['counts']['WBC'])
            platelet_counts.append(data[key]['counts']['Platelets'])
    return xml_data, rbc_counts, wbc_counts, platelet_counts

def compute_loss(cell_count, train_idx, test_idx):
    """
    Compute a loss that measures how different the count distributions are.
    We want this loss to be as small as possible.
    
    Higher Wasserstein distances = more different distributions = worse.
    We therefore use the Wasserstein distance directly as a loss (lower is better).
    """
    
    train_loss = 0.0
    test_loss = 0.0

    def pair_loss(counts, split_idx):
        #a here is the full array of counts (the ground truth distribution)
        # b is the counts for the split we are comparing against

        a = np.asarray(counts).copy()
        b = np.asarray(counts)[split_idx]
        return wasserstein_distance(a, b)   # already a distance → lower is better
    
    train_loss += pair_loss(cell_count, train_idx)
    test_loss += pair_loss(cell_count, test_idx)
    return train_loss, test_loss

def optimize_split(cell_count_rb, cell_count_wb, cell_count_pl,
                   train_idx, test_idx, n_iterations=9000, verbose=True, only_idx = False):
    """
    Hill-climbing optimizer.
    
    Starts from the given initial split and repeatedly tries random swaps.
    Only accepts a swap if it reduces the loss.
    """
    # Make local copies so we do not modify the original lists
    train_idx = list(train_idx)
    test_idx  = list(test_idx)
    cell_count_rb = np.asarray(cell_count_rb).copy()
    cell_count_wb = np.asarray(cell_count_wb).copy()
    cell_count_pl = np.asarray(cell_count_pl).copy()

    current_train_loss_rb, current_test_loss_rb = compute_loss(cell_count_rb, train_idx, test_idx)
    current_train_loss_wb, current_test_loss_wb = compute_loss(cell_count_wb, train_idx, test_idx)
    current_train_loss_pl, current_test_loss_pl = compute_loss(cell_count_pl, train_idx, test_idx)

    current_combined_loss_train = current_train_loss_rb + current_train_loss_wb + current_train_loss_pl
    current_combined_loss_test = current_test_loss_rb + current_test_loss_wb + current_test_loss_pl
    current_combined_loss = current_combined_loss_train + current_combined_loss_test

    all_train_losses_rb = [current_train_loss_rb]
    all_test_losses_rb = [current_test_loss_rb]
    all_train_losses_wb = [current_train_loss_wb]
    all_test_losses_wb = [current_test_loss_wb]
    all_train_losses_pl = [current_train_loss_pl]
    all_test_losses_pl = [current_test_loss_pl]
    all_train_losses = [current_combined_loss_train]
    all_test_losses = [current_combined_loss_test]

    if verbose:
        # print(f"Initial RBC train loss: {current_train_loss_rb:.4f}")
        # print(f"Initial RBC test loss: {current_test_loss_rb:.4f}")
        # print(f"Initial WBC train loss: {current_train_loss_wb:.4f}")
        # print(f"Initial WBC test loss: {current_test_loss_wb:.4f}")
        # print(f"Initial PL train loss: {current_train_loss_pl:.4f}")
        # print(f"Initial PL test loss: {current_test_loss_pl:.4f}")
        print()
        print(f"Initial train loss: {current_combined_loss_train:.4f}")
        print(f"Initial test loss: {current_combined_loss_test:.4f}")
        print(f"Initial Combined Loss: {current_combined_loss:.4f}")
        print("START")

    for iteration in tqdm(range(n_iterations)):
        # Randomly pick one image from each of the two sets
        i = np.random.randint(len(cell_count_rb[train_idx]))
        j = np.random.randint(len(cell_count_rb[test_idx]))
        

        # Remember the original image IDs so we can undo the swap if needed
        # Correct version
        idx_a = train_idx[i]          # actual image index
        idx_b = test_idx[j]           # actual image index

        # Swap the indices
        train_idx[i] = idx_b
        test_idx[j] = idx_a

        new_train_loss_rb, new_test_loss_rb = compute_loss(cell_count_rb, train_idx, test_idx)
        new_train_loss_wb, new_test_loss_wb = compute_loss(cell_count_wb, train_idx, test_idx)
        new_train_loss_pl, new_test_loss_pl = compute_loss(cell_count_pl, train_idx, test_idx)

        new_combined_loss_train = new_train_loss_rb + new_train_loss_wb + new_train_loss_pl
        new_combined_loss_test = new_test_loss_rb + new_test_loss_wb + new_test_loss_pl
        new_combined_loss = new_combined_loss_train + new_combined_loss_test

        
        
        if new_combined_loss < (current_combined_loss):
            # Improvement! Keep the swap and update the current loss
            current_train_loss_rb = new_train_loss_rb
            current_test_loss_rb = new_test_loss_rb
            current_train_loss_wb = new_train_loss_wb
            current_test_loss_wb = new_test_loss_wb
            current_train_loss_pl = new_train_loss_pl
            current_test_loss_pl = new_test_loss_pl

            current_combined_loss_train = new_combined_loss_train
            current_combined_loss_test = new_combined_loss_test
            current_combined_loss = new_combined_loss

            all_train_losses_rb.append(current_train_loss_rb)
            all_test_losses_rb.append(current_test_loss_rb)
            all_train_losses_wb.append(current_train_loss_wb)
            all_test_losses_wb.append(current_test_loss_wb)
            all_train_losses_pl.append(current_train_loss_pl)
            all_test_losses_pl.append(current_test_loss_pl)
            all_train_losses.append(current_combined_loss_train)
            all_test_losses.append(current_combined_loss_test)

        else:
            # Revert the swap
            train_idx[i] = idx_a
            test_idx[j] = idx_b
            all_train_losses_rb.append(current_train_loss_rb)
            all_test_losses_rb.append(current_test_loss_rb)
            all_train_losses_wb.append(current_train_loss_wb)
            all_test_losses_wb.append(current_test_loss_wb)
            all_train_losses_pl.append(current_train_loss_pl)
            all_test_losses_pl.append(current_test_loss_pl)
            all_train_losses.append(current_combined_loss_train)
            all_test_losses.append(current_combined_loss_test)
    if verbose:
        print()
        print(f"Final Train Loss: {current_combined_loss_train:.4f}")
        print(f"Final Test Loss: {current_combined_loss_test:.4f}")
        print(f"Final Combined Loss: {current_combined_loss:.4f}")

    if only_idx:
        return train_idx, test_idx
    else:
        return train_idx, test_idx, all_train_losses_rb,\
            all_test_losses_rb, all_train_losses_wb, all_test_losses_wb,\
            all_train_losses_pl, all_test_losses_pl, all_train_losses, all_test_losses