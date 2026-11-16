import torch

ROWS_PER_BLOCK = 64


def franky_reader_row_blocks(
    rows_per_request: list[int], *, device: torch.device
) -> torch.Tensor:
    """
    Args:
        rows_per_request: each request's rows, laid out request after request.
    Returns:
        [blocks, 3] int32: the first row, end row and request of each block of
        at most ``ROWS_PER_BLOCK`` rows of one request.
    """
    blocks = []
    first_row = 0
    for request, num_rows in enumerate(rows_per_request):
        end_row = first_row + num_rows
        for start in range(first_row, end_row, ROWS_PER_BLOCK):
            blocks.append((start, min(start + ROWS_PER_BLOCK, end_row), request))
        first_row = end_row
    table = torch.tensor(blocks, dtype=torch.int32).reshape(-1, 3)
    if device.type == "cpu":
        return table
    return table.pin_memory().to(device, non_blocking=True)
