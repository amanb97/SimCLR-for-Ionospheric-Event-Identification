import numpy as np

def augment_power(
    data,
    negative_value=-9999,
    noise_strength=0.2,
    scale_range=(0.9, 1.1),
    max_shift=2,
    max_removed_points=25,
    swap_prob=0.1,
    mask_prob=0.1,
    saturation_factor=1.2,
    augment_probabilities=None,
    verbose=False
):
    """
    Apply augmentation strategies to radar power data, including noise addition,
    scaling, translation, swapping, masking, and saturation adjustment.

    """

    data_power = data.copy()

    #define default probabilities if none provided
    if augment_probabilities is None:
        augment_probabilities = {
            'add_noise': 0.5,
            'scale_data': 0.5,
            'translate_y': 0.5,
            'translate_x': 0.5,
            'swap_adjacent_range_gates': 0.5,
            'mask_data': 0.5,
            'adjust_saturation': 0.5
        }
    #define augmentation functions
    def add_noise(data):
        valid_mask = data != negative_value
        if np.any(valid_mask):  
            data_range = data[valid_mask].max() - data[valid_mask].min()
            dynamic_noise_strength = noise_strength * data_range
            noise = np.random.normal(0, dynamic_noise_strength, size=data.shape)
            data[valid_mask] += noise[valid_mask]
        return data

    def scale_data_func(data):
        valid_mask = data != negative_value
        scale_factor = np.random.uniform(*scale_range)
        data[valid_mask] *= scale_factor
        return data

    def translate_y(data):
        attempt = 0
        ymax_shift = min(max_shift * 2, data.shape[1] - 1)
        while attempt < 3:  
            shift = np.random.randint(-ymax_shift, ymax_shift + 1)
            if shift == 0:
                return data  

            num_time_steps, num_range_gates = data.shape
            shifted_data = np.full_like(data, negative_value)

            if shift > 0:
                shifted_data[:, shift:] = data[:, :-shift]
            else:
                shifted_data[:, :num_range_gates + shift] = data[:, -shift:]

            removed_points = np.sum((data != negative_value) & (shifted_data == negative_value))
            if removed_points <= max_removed_points:
                if verbose:
                    print(f"Applied translate_y with shift {shift}")
                    print(f"removed {removed_points} from total {num_time_steps * num_range_gates}" )
                return shifted_data  
            attempt += 1

        if verbose:
            print(f"Translation in Y aborted after {attempt} attempts: {removed_points} points would be removed.")
        return data  

    def translate_x(data):
        attempt = 0
        while attempt < 3:  
            limit = min(max_shift, data.shape[0] - 1)
            shift = np.random.randint(-limit, limit + 1)
            if shift == 0:
                return data  

            num_time_steps, num_range_gates = data.shape
            shifted_data = np.full_like(data, negative_value)

            if shift > 0:

                shifted_data[shift:, :] = data[:-shift, :]
            else:
                shifted_data[:num_time_steps + shift, :] = data[-shift:, :]

            removed_points = np.sum((data != negative_value) & (shifted_data == negative_value))
            if removed_points <= max_removed_points:
                if verbose:
                    print(f"Applied translate_x with shift {shift}")
                return shifted_data  
            attempt += 1

        if verbose:
            print(f"Translation in X aborted after {attempt} attempts: {removed_points} points would be removed.")
        return data

    def swap_adjacent_range_gates(data):
        num_time_steps, num_range_gates = data.shape
        for t in range(num_time_steps):
            for r in range(num_range_gates - 1):
                if np.random.rand() < swap_prob:
                    if (data[t, r] != negative_value) and (data[t, r + 1] != negative_value):
                        data[t, r], data[t, r + 1] = data[t, r + 1], data[t, r]
        if verbose:
            print(f"Applied swap_adjacent_range_gates with swap_prob {swap_prob}")
        return data

    def mask_data(data):
        num_time_steps, num_range_gates = data.shape
        mask = (np.random.rand(num_time_steps, num_range_gates) < mask_prob) & (data != negative_value)
        data[mask] = negative_value
        if verbose:
            print(f"Applied mask_data with mask_prob {mask_prob}")
        return data
    
    
    def adjust_saturation(data):
        valid_mask = data != negative_value
        if not np.any(valid_mask):
            return data
        mean_value = np.mean(data[valid_mask])
        data[valid_mask] = mean_value + saturation_factor * (data[valid_mask] - mean_value)
        if verbose:
            print(f"Applied adjust_saturation with saturation_factor {saturation_factor}")
        return data

    augmentations = {
        'add_noise': add_noise,
        'scale_data': scale_data_func,
        'translate_y': translate_y,
        'translate_x': translate_x,
        'swap_adjacent_range_gates': swap_adjacent_range_gates,
        'mask_data': mask_data,
        'adjust_saturation': adjust_saturation
    }

    #apply augmentations based on probabilities
    for aug_name, aug_func in augmentations.items():
        prob = augment_probabilities.get(aug_name, 0)
        if np.random.rand() < prob:
            data_power = aug_func(data_power)
            if verbose and aug_name not in ['translate_y', 'translate_x']:
                print(f"Applied augmentation: {aug_name}")
        else:
            if verbose:
                print(f"Skipped augmentation: {aug_name}")

    return data_power
