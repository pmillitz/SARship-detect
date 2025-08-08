# test_sar_slc_augmentation.py

import numpy as np
import pytest
import matplotlib.pyplot as plt
import os
from sar_slc_augmentation import SARSLCPreprocessingAugmentation


@pytest.fixture
def sample_data():
    h, w = 100, 100
    real = np.random.randn(h, w).astype(np.float32)
    imag = np.random.randn(h, w).astype(np.float32)
    data = real + 1j * imag

    labels = np.array([
        [0, 0.25, 0.25, 0.2, 0.2],
        [1, 0.75, 0.25, 0.2, 0.2],
        [0, 0.25, 0.75, 0.2, 0.2],
        [1, 0.75, 0.75, 0.2, 0.2],
    ], dtype=np.float32)
    return data.astype(np.complex64), labels


def test_horizontal_flip(sample_data):
    data, labels = sample_data
    aug = SARSLCPreprocessingAugmentation()
    hflip = aug._create_complex_hflip()
    flipped_data, flipped_labels = hflip(data.copy(), labels.copy())

    expected_x = 1.0 - labels[:, 1]
    np.testing.assert_allclose(flipped_labels[:, 1], expected_x, rtol=1e-5)


def test_vertical_flip(sample_data):
    data, labels = sample_data
    aug = SARSLCPreprocessingAugmentation()
    vflip = aug._create_complex_vflip()
    flipped_data, flipped_labels = vflip(data.copy(), labels.copy())

    expected_y = 1.0 - labels[:, 2]
    np.testing.assert_allclose(flipped_labels[:, 2], expected_y, rtol=1e-5)


def test_rotation_90(sample_data):
    data, labels = sample_data
    aug = SARSLCPreprocessingAugmentation()
    rotate = aug._create_complex_rotate()

    import random
    old_randint = random.randint
    random.randint = lambda a, b: 1  # force 90 degrees

    rotated_data, rotated_labels = rotate(data.copy(), labels.copy())
    random.randint = old_randint

    expected_x = labels[:, 2]  # new_x = old_y
    expected_y = 1.0 - labels[:, 1]  # new_y = 1 - old_x
    np.testing.assert_allclose(rotated_labels[:, 1], expected_x, rtol=1e-5)
    np.testing.assert_allclose(rotated_labels[:, 2], expected_y, rtol=1e-5)


def test_translation(sample_data):
    data, labels = sample_data
    aug = SARSLCPreprocessingAugmentation(min_visibility=0.0)
    translate = aug._create_complex_translate()

    import random
    old_randint = random.randint
    call_counter = [0]

    def fixed_randint(a, b):
        call_counter[0] += 1
        return 10 if call_counter[0] == 1 else 5  # shift_x=10, shift_y=5

    random.randint = fixed_randint
    translated_data, translated_labels = translate(data.copy(), labels.copy())
    random.randint = old_randint

    expected_shift_x = 10 / 100
    expected_shift_y = 5 / 100
    shifted_x = labels[:, 1] + expected_shift_x
    shifted_y = labels[:, 2] + expected_shift_y

    np.testing.assert_allclose(translated_labels[:, 1], shifted_x, rtol=1e-2)
    np.testing.assert_allclose(translated_labels[:, 2], shifted_y, rtol=1e-2)


def test_phase_shift(sample_data):
    data, _ = sample_data
    aug = SARSLCPreprocessingAugmentation()
    phase_shift = aug._create_phase_shift()
    shifted = phase_shift(data)

    phase_diff = np.angle(shifted[0, 0]) - np.angle(data[0, 0])
    assert -np.pi/4 <= phase_diff <= np.pi/4


def test_complex_speckle_noise_addition(sample_data):
    data, _ = sample_data
    aug = SARSLCPreprocessingAugmentation()
    speckle = aug._create_complex_speckle()
    noisy = speckle(data)

    diff = np.abs(noisy - data)
    assert np.mean(diff) > 0.01


def test_gaussian_filter_smooths(sample_data):
    data, _ = sample_data
    aug = SARSLCPreprocessingAugmentation()
    filt = aug._create_complex_gaussian_filter()
    smooth = filt(data)

    orig_std = np.std(np.abs(data))
    filt_std = np.std(np.abs(smooth))
    assert filt_std < orig_std
    assert np.isclose(np.angle(data[0, 0]), np.angle(smooth[0, 0]), atol=1e-2)


def test_mosaic():
    aug = SARSLCPreprocessingAugmentation()
    h, w = 64, 64
    test_images_labels = []

    for i in range(4):
        data = (np.ones((h, w)) * (i + 1)).astype(np.complex64)
        label = np.array([[i % 2, 0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
        test_images_labels.append((data, label))

    mosaic_data, mosaic_labels = aug.create_mosaic(test_images_labels, output_size=(128, 128))

    assert mosaic_data.shape == (128, 128)
    assert mosaic_labels.shape[1] == 5
    assert np.all(mosaic_labels[:, 1:] >= 0.0)
    assert np.all(mosaic_labels[:, 1:] <= 1.0)


def test_chained_augmentations(sample_data):
    data, labels = sample_data
    aug = SARSLCPreprocessingAugmentation(
        geometric_probs={0: {'hflip': 1.0, 'vflip': 1.0, 'rotate': 1.0, 'translate': 1.0}},
        sar_probs={0: {'phase_shift': 1.0, 'amplitude_scale': 0.0, 'complex_speckle': 0.0, 'gaussian_filter': 0.0}}
    )
    labels[:, 0] = 0

    aug_result = aug.augment_single(data.copy(), labels.copy(), force_augment=True)[0]
    aug_data, aug_labels, desc = aug_result

    assert isinstance(aug_data, np.ndarray)
    assert isinstance(aug_labels, np.ndarray)
    assert 'hflip' in desc and 'vflip' in desc and 'rotate' in desc and 'translate' in desc and 'phase_shift' in desc


def test_visualize_augmented_output(tmp_path, sample_data):
    data, labels = sample_data
    aug = SARSLCPreprocessingAugmentation(
        geometric_probs={0: {'hflip': 1.0, 'vflip': 0.0, 'rotate': 0.0, 'translate': 0.0}},
        sar_probs={0: {'phase_shift': 0.0, 'amplitude_scale': 0.0, 'complex_speckle': 0.0, 'gaussian_filter': 0.0}}
    )
    labels[:, 0] = 0
    aug_data, aug_labels, desc = aug.augment_single(data, labels, force_augment=True)[0]

    mag = np.abs(aug_data)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mag, cmap='gray')

    H, W = mag.shape
    for cls, xc, yc, w, h in aug_labels:
        x0 = (xc - w/2) * W
        y0 = (yc - h/2) * H
        width = w * W
        height = h * H
        rect = plt.Rectangle((x0, y0), width, height, edgecolor='lime', facecolor='none', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x0, y0 - 2, f"{int(cls)}", color='lime', fontsize=8)

    ax.set_title(f"Augmented: {desc}")
    ax.axis('off')

    #output_path = os.path.join(tmp_path, f"visual_{desc}.png")
    output_path = "./"
    fig.savefig(output_path)
    plt.close(fig)

    assert os.path.exists(output_path)

