import torch


def get_last_token(output, mask):
    mask_max = (~mask).max(dim=1)
    lengths = mask_max.indices
    lengths[~mask_max.values] = mask.shape[1]
    lengths -= 1

    # expand to match the shape of all_token_embeddings
    indices = lengths.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
    last_token = torch.gather(output, dim=1, index=indices).squeeze(1)
    return last_token
