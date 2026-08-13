"""Honesty and geometry contracts for the expert-atlas animation."""

import numpy as np


def test_phase_luminance_uses_one_shared_scale():
    from llmopt.figures.atlas_visuals import phase_luminance

    prefill = np.array([[0.0, 0.5, 1.0]])
    decode = np.array([[0.0, 0.25, 0.5]])
    pre_lum, dec_lum = phase_luminance(prefill, decode, gamma=1.0)

    np.testing.assert_allclose(pre_lum, prefill)
    np.testing.assert_allclose(dec_lum, decode)


def test_focus_pair_is_stable_and_preserves_layer():
    from llmopt.figures.atlas_visuals import deterministic_focus_pair

    carriers = np.array([[4, 20], [0, 35], [0, 10]])
    controls = np.array([[4, 23], [0, 37], [0, 13]])

    carrier, control = deterministic_focus_pair(carriers, controls)

    assert carrier == (0, 10)
    assert control == (0, 13)


def test_block_field_keeps_gutters_dark():
    from llmopt.figures.atlas_visuals import block_field

    luminance = np.ones((2, 3))
    ramp = ["#000000", "#ffffff"]
    image = block_field(luminance, ramp, "#101010", cell=2, gutter=1)

    assert image.shape == (6, 9, 3)
    np.testing.assert_array_equal(image[0:2, 0:2], 255)
    np.testing.assert_array_equal(image[2, :], 16)
    np.testing.assert_array_equal(image[:, 2], 16)


def test_perspective_projection_preserves_cell_count_and_depth_order():
    from llmopt.figures.atlas_visuals import perspective_projection

    points = perspective_projection(3, 4, width=800, height=450)

    assert points.shape == (3, 4, 2)
    assert points[0, 0, 1] < points[1, 0, 1] < points[2, 0, 1]
    assert np.ptp(points[0, :, 0]) < np.ptp(points[2, :, 0])
    np.testing.assert_allclose(points[:, :, 0].mean(axis=1), 400)


def test_rail_fraction_is_anchored_at_zero():
    """Guards the truncated-axis defect: an origin chosen near the data
    inflated 244-vs-189 to a ~3.9x visual ratio against a true 1.29x."""
    from llmopt.figures.atlas_visuals import rail_fraction

    full = rail_fraction(189, 360)
    matched = rail_fraction(217, 360)
    carriers = rail_fraction(244, 360)

    assert full == 189 / 360
    assert carriers / full == 244 / 189
    assert matched / full == 217 / 189
    assert rail_fraction(0, 360) == 0.0
