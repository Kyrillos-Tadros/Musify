import numpy as np

def augment_data(X_train, y_train, num_augmented_samples=8000):
    """
    Augment the training data using various techniques suitable for audio features.
    """
    X_augmented = []
    y_augmented = []

    # Append original samples to X_augmented and y_augmented
    X_augmented.extend(X_train)
    y_augmented.extend(y_train)

    # Generate and append augmented samples to X_augmented and y_augmented
    for _ in range(num_augmented_samples):
        # Select a random sample from the training data
        idx = np.random.randint(0, len(X_train))
        sample = X_train[idx].copy()
        label = y_train[idx]

        # Apply augmentation techniques...
        # Random scaling
        scale_factor = np.random.uniform(0.9, 1.1)
        sample *= scale_factor

        # Random noise
        noise_factor = np.random.uniform(-0.01, 0.01)
        sample += noise_factor

        # Time shifting
        shift_factor = np.random.uniform(-0.1, 0.1)
        sample = np.roll(sample, int(len(sample) * shift_factor), axis=0)

        # Feature masking
        mask_ratio = np.random.uniform(0.0, 0.2)
        mask_length = int(len(sample) * mask_ratio)
        mask_start = np.random.randint(0, len(sample) - mask_length)
        sample[mask_start:mask_start + mask_length] = 0.0

        # Mixup
        if np.random.uniform() < 0.5:
            idx2 = np.random.randint(0, len(X_train))
            sample2 = X_train[idx2]
            label2 = y_train[idx2]
            mixup_factor = np.random.uniform(0, 1)
            sample = mixup_factor * sample + (1 - mixup_factor) * sample2
            label = mixup_factor * label + (1 - mixup_factor) * label2

        # Normalize the augmented sample
        augmented_sample = (sample - np.mean(sample)) / np.std(sample)

        # Append the augmented sample and label
        X_augmented.append(augmented_sample)
        y_augmented.append(label)

    # Determine the input_shape after augmentation
    input_shape = X_augmented[0].shape

    return np.array(X_augmented), np.array(y_augmented)