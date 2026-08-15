import numpy as np
from scipy.stats import wasserstein_distance
from tqdm import tqdm

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
                   train_idx, test_idx, n_iterations=9000, verbose=True):
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
    print()
    print(f"Final Train Loss: {current_combined_loss_train:.4f}")
    print(f"Final Test Loss: {current_combined_loss_test:.4f}")
    print(f"Final Combined Loss: {current_combined_loss:.4f}")
    return train_idx, test_idx, all_train_losses_rb,\
        all_test_losses_rb, all_train_losses_wb, all_test_losses_wb,\
        all_train_losses_pl, all_test_losses_pl, all_train_losses, all_test_losses