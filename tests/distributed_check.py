"""Run with torchrun --standalone --nproc-per-node=2; no dataset or Qwen needed."""

import argparse
import copy
from pathlib import Path
import torch
import torch.distributed as dist
from cedr.balance import PromptMSEBalance
from cedr.common import distributed_setup, average_gradients, atomic_json, seed_all
from cedr.objectives import nft_numerator
from cedr.optim import muon_with_aux_adam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rank, world, device = distributed_setup()
    seed_all(42)
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0], device=device))
    balance = PromptMSEBalance(target_ratio=2.0)
    downstream = torch.tensor([3.0 + rank, 4.0 - rank], device=device)
    aux = torch.tensor([rank, 2.0 + rank], device=device)
    for _ in range(2):
        (p.dot(downstream) / 2).backward()
        balance.accumulate(p.dot(aux), [p], 2)
    result = balance.combine([p], 2)
    d = torch.tensor([3.0 + (world - 1) / 2, 4.0 - (world - 1) / 2], device=device)
    m = torch.tensor([(world - 1) / 2, 2.0 + (world - 1) / 2], device=device)
    expected = d + 2 * d.norm() / m.norm() * m
    torch.testing.assert_close(p.grad, expected)
    saved = balance.state_dict()
    resume = PromptMSEBalance(target_ratio=2.0)
    resume.load_state_dict(saved)
    assert resume.state_dict() == saved
    # Unequal retained token counts; reference gradient is formed over all ranks.
    q = torch.nn.Linear(1, 1, bias=False).to(device)
    q.weight.data.fill_(0.3)
    count = rank + 1
    mask = torch.ones(count, 3, dtype=torch.bool, device=device)
    mask[-1, -1] = False
    denom = mask.sum().float()
    dist.all_reduce(denom)
    x = q.weight.reshape(1, 1, 1).expand(count, 3, 2)
    zero = torch.zeros_like(x)
    rho = torch.full((count,), 0.7, device=device)
    loss = nft_numerator(x, zero, zero, zero, rho, mask) * (world / denom)
    loss.backward()
    average_gradients(q)
    expected_grad = 2 * 0.3 * (5 + 0.1)
    torch.testing.assert_close(q.weight.grad, torch.full_like(q.weight, expected_grad))
    # Partial prompt batches normalize by REAL global rows, not padded ranks.
    local_real = 3 if rank == 0 else 1
    global_real = 3 + (world - 1)
    param = torch.nn.Linear(1, 1, bias=False).to(device)
    param.weight.data.fill_(1.0)
    (param.weight.sum() * (rank + 1) * world * local_real / global_real).backward()
    average_gradients(param)
    reference = (3 + sum(r + 1 for r in range(1, world))) / global_real
    torch.testing.assert_close(
        param.weight.grad, torch.full_like(param.weight, reference)
    )
    # Replicated optimizer updates must equal updates from averaged gradients.
    model = torch.nn.Linear(3, 2).to(device)
    control = copy.deepcopy(model)
    optimizer = muon_with_aux_adam(model, 0.002)
    control_optimizer = muon_with_aux_adam(control, 0.002)
    for step in range(3):
        for parameter, target in zip(model.parameters(), control.parameters()):
            pattern = torch.arange(
                parameter.numel(), device=device, dtype=parameter.dtype
            ).reshape_as(parameter)
            parameter.grad = pattern + rank + step + 1
            target.grad = pattern + (world - 1) / 2 + step + 1
        average_gradients(model)
        optimizer.step()
        control_optimizer.step()
        for parameter, target in zip(model.parameters(), control.parameters()):
            torch.testing.assert_close(parameter, target, rtol=0, atol=0)
        if step == 1:
            state = copy.deepcopy(optimizer.state_dict())
            optimizer = muon_with_aux_adam(model, 0.002)
            optimizer.load_state_dict(state)
    for parameter in model.parameters():
        rank_zero = parameter.detach().clone()
        dist.broadcast(rank_zero, src=0)
        torch.testing.assert_close(parameter, rank_zero, rtol=0, atol=0)
    if rank == 0:
        atomic_json(
            dict(
                status="passed",
                world_size=world,
                backend=dist.get_backend(),
                balanced_gradient=p.grad.tolist(),
                nft_gradient=float(q.weight.grad),
                tail_gradient=float(param.weight.grad),
                replicated_optimizer="passed",
            ),
            Path(args.output) / "distributed.json",
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
