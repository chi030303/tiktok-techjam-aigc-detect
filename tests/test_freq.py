# 2026-08-29, tianqi, unit tests for FFT / high-pass views
from PIL import Image

from src.data.freq import apply_input_mode, fft_mag_rgb, highpass_residual


def test_freq_views_keep_rgb_size():
    img = Image.new("RGB", (32, 24), (10, 20, 30))
    hp = highpass_residual(img)
    fft = fft_mag_rgb(img)
    assert hp.size == img.size and hp.mode == "RGB"
    assert fft.size == img.size and fft.mode == "RGB"
    assert apply_input_mode(img, "rgb").size == img.size
    assert apply_input_mode(img, "fft").mode == "RGB"
# end
