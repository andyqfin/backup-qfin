from tqdm import tqdm

def show_progress(batch_size, full_size):

    total_batches = full_size // batch_size
    progress_bar = tqdm(total=total_batches, desc="Processing", unit="batch")

    return total_batches, progress_bar