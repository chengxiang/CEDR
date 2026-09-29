# Third-party notices

- **ELF**: architecture and training kernels derive from [lillian039/ELF](https://github.com/lillian039/ELF). The upstream MIT notice is retained in the repository's `LICENSE`.
- **Qwen3**: [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), pinned revision recorded in configurations. Model/tokenizer artifacts are obtained separately and retain their own license.
- **Muon**: `muon-optimizer==0.1.0` supplies the optimizer container. The FP32 Newton–Schulz, bias-corrected momentum, Nesterov auxiliary Adam, and parameter-layout scaling patches are retained in `cedr.optim`.
- **DiffusionNFT**: the NFT objective follows [NVLabs/DiffusionNFT](https://github.com/NVlabs/DiffusionNFT), revision `cbb14f84b8312b620390dfcbe2fab69c1383b104`. CEDR uses gold anchoring, tiered rewards, reference regularization, and group filtering with synchronous training clocks. No NVLabs weights are bundled.
- **Math-Verify**: used as an external canonical math scorer, pinned to 0.9.0.
- **EvalPlus**: official external code scorer and sanitizer, pinned to `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`. No tests or benchmark data are redistributed here.

Data-source and model licenses apply independently of this code's MIT license. Downloaded assets and installed dependency packages retain their original notices.
