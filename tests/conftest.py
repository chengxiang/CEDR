import copy
from pathlib import Path
import pytest
import torch
import yaml


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def cfg():
    c = yaml.safe_load((Path(__file__).parents[1] / "configs/gsm8k_b.yaml").read_text())
    c["model"].update(
        text_encoder_dim=12,
        max_length=16,
        hidden_size=32,
        depth=2,
        num_heads=4,
        bottleneck_dim=8,
        decoder_dim=8,
        vocab_size=64,
    )
    c["prompt"].update(
        vocab_size=64,
        max_length=16,
        external_embedding_dim=16,
        bottleneck_dim=8,
        hidden_size=32,
        depth=2,
        num_heads=4,
        output_dim=12,
    )
    c["representation"].update(layers=[1, 2, 3], dimensions=[4, 4, 4])
    c["data"].update(max_tokens=16, max_prompt_tokens=8, pad_token_id=0)
    c["training"].update(
        effective_batch=4, micro_batch=2, bf16=False, compile=False, save_every=1
    )
    c["stages"].update(flow_epochs=2, prompt_epochs=2, joint_epochs=2, nft_updates=1)
    c["nft"].update(
        prompt_groups=2, generated_per_group=2, draws_per_endpoint=1, rollout_steps=2
    )
    return c


@pytest.fixture
def rows():
    return [
        dict(
            id=str(i),
            prompt_id=str(i),
            input_ids=[1, 2, 3, 4, 5, 6],
            prompt_length=3,
            content_end=5,
            prompt="one plus one",
            answer="2",
            weight=1.0,
            metadata={},
            physical_index=i,
        )
        for i in range(8)
    ]
