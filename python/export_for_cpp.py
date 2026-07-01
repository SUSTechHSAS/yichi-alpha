"""
Export a model checkpoint to a C++-loadable TorchScript format.

C++ LibTorch's torch::jit::load() requires TorchScript format (not Python pickled state_dict).
This script uses torch.jit.script() to convert the model to TorchScript and saves it.

Also embeds the model config (board_size, channels, n_blocks) as TorchScript attributes
so the C++ side can reconstruct the model with matching dimensions.
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, '/home/z/my-project/download/yichi-alpha/python')
from model import load_checkpoint


def export_for_cpp(python_ckpt_path: str, cpp_ckpt_path: str):
    """Load Python checkpoint, save as TorchScript for C++."""
    model = load_checkpoint(python_ckpt_path, device='cpu')
    model.eval()

    # Embed config as extra files metadata
    extra_files = {
        'board_size.txt': str(model.board_size),
        'in_channels.txt': str(model.in_channels),
        'channels.txt': str(model.channels),
        'n_blocks.txt': str(model.n_blocks),
    }

    # Convert to TorchScript
    try:
        scripted = torch.jit.script(model)
        # Use torch.jit.save (not module.save) to support extra_files
        torch.jit.save(scripted, cpp_ckpt_path, extra_files)
        print(f"Exported (TorchScript): {python_ckpt_path} → {cpp_ckpt_path}")
    except Exception as e:
        print(f"torch.jit.script failed: {e}")
        print("Falling back to torch.jit.trace...")
        dummy = torch.randn(1, 11, model.board_size, model.board_size)
        traced = torch.jit.trace(model, dummy)
        torch.jit.save(traced, cpp_ckpt_path, extra_files)
        print(f"Exported (Traced): {python_ckpt_path} → {cpp_ckpt_path}")

    print(f"  Config: board_size={model.board_size}, in_channels={model.in_channels}, "
          f"channels={model.channels}, n_blocks={model.n_blocks}")

    # Verify it loads back
    loaded = torch.jit.load(cpp_ckpt_path)
    dummy = torch.randn(1, 11, model.board_size, model.board_size)
    p1, v1 = model(dummy)
    p2, v2 = loaded(dummy)
    diff_p = (p1 - p2).abs().max().item()
    diff_v = (v1 - v2).abs().max().item()
    print(f"  Verification: max policy diff = {diff_p:.2e}, max value diff = {diff_v:.2e}")
    assert diff_p < 1e-5 and diff_v < 1e-5, "TorchScript conversion has significant diff!"
    print(f"  ✓ Conversion verified (diffs < 1e-5)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        ckpt_dir = Path('/home/z/my-project/download/yichi-alpha/checkpoints')
        for p in sorted(ckpt_dir.glob('model_iter*.pt')):
            if p.name.endswith('_cpp.pt'):
                continue
            cpp_path = str(p).replace('.pt', '_cpp.pt')
            export_for_cpp(str(p), cpp_path)
    else:
        export_for_cpp(sys.argv[1], sys.argv[2])


