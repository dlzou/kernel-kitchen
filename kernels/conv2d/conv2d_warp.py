from typing import Any

import warp as wp


WP_DEVICE = wp.get_device()

@wp.kernel
def _conv2d_basic(
    input_tensor: wp.array4d(dtype=Any),
    weight: wp.array4d(dtype=Any),
    output_tensor: wp.array4d(dtype=Any),
    stride: int,
    padding: int,
):
    idy, idx = wp.tid()
    top_pady = wp.max(padding - idy * stride, 0)
    bottom_pady = wp.max(idy * stride + weight.shape[2] - (padding + input_tensor.shape[2]), 0)
    left_padx = wp.max(padding - idx * stride, 0)
    right_padx = wp.max(idx * stride + weight.shape[3] - (padding + input_tensor.shape[3]), 0)

    for n in range(input_tensor.shape[0]):
        for c in range(weight.shape[0]):
            for d in range(weight.shape[1]):
                for h in range(top_pady, weight.shape[2] - bottom_pady):
                    for w in range(left_padx, weight.shape[3] - right_padx):
                        ih = wp.max(idy * stride - padding, 0) - top_pady + h
                        iw = wp.max(idx * stride - padding, 0) - left_padx + w
                        output_tensor[n, c, idy, idx] += input_tensor[n, d, ih, iw] * weight[c, d, h, w]


def conv2d_basic_setup(
    input_tensor: wp.array(dtype=Any),
    weight: wp.array(dtype=Any),
    stride = 1,
    padding = 0,
) -> wp.array(dtype=Any):
    """
    Arguments:
        input_tensor: (batches, in_channels, input_height, input_width)
        weight: (out_channels, in_channels, kernel_height, kernel_width)
    Returns:
        Tensor with conv2d applied
    """
    assert len(input_tensor.shape) == 4 and len(weight.shape) == 4, "input_tensor and kernel must be 4D"
    assert input_tensor.shape[1] == weight.shape[1], "input_tensor and kernel must have equal number of in_channels"

    n = input_tensor.shape[0]
    d = weight.shape[0]
    h = (input_tensor.shape[-2] - weight.shape[-2] + 2 * padding) // stride + 1
    w = (input_tensor.shape[-1] - weight.shape[-1] + 2 * padding) // stride + 1
    output_tensor = wp.zeros((n, d, h, w), dtype=input_tensor.dtype, device=WP_DEVICE)

    def conv2d_basic():
        wp.launch(
            _conv2d_basic,
            dim=(h, w),
            inputs=[input_tensor, weight, output_tensor, stride, padding],
        )
        return output_tensor

    return conv2d_basic
