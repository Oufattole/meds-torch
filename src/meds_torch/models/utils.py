import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        """
        Performs an all_gather operation that allows gradients to flow.

        Args:
            tensor (Tensor): The tensor to gather from all devices.

        Returns:
            Tensor: Concatenated tensor from all devices.
        """
        world_size = dist.get_world_size()
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, tensor)
        ctx.world_size = world_size
        ctx.local_tensor_shape = tensor.shape
        return torch.cat(gathered_tensors, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Handles backpropagation for the all_gather operation.

        Args:
            grad_output (Tensor): Gradients passed from the next layer.

        Returns:
            Tensor: Gradient for the local portion of the tensor.
        """
        # Split the gradient into chunks for each device
        grad_chunks = torch.chunk(grad_output, ctx.world_size, dim=0)
        # Return the gradient for the local tensor
        local_grad = grad_chunks[dist.get_rank()]
        return local_grad
