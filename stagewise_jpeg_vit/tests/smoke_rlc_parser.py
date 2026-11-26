import io

from PIL import Image
import torch  # noqa: F401 - required before byte_manip so libc10 is loaded


def main():
    try:
        import byte_manip as bm
    except ImportError:
        print("SKIP byte_manip unavailable")
        return

    if not hasattr(bm, "read_syntax_patches_from_buffer"):
        print("SKIP read_syntax_patches_from_buffer unavailable")
        return

    image = Image.new("RGB", (224, 224), color=(128, 64, 32))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    syntax = bm.read_syntax_patches_from_buffer(buffer.getvalue())
    mcus = syntax["mcus"]
    assert len(mcus) > 0
    dc_count = sum(1 for mcu in mcus for block in mcu["blocks"] for token in block["tokens"] if token["token_type"] == "DC")
    assert dc_count > 0
    print(f"OK rlc parser mcus={len(mcus)} dc_events={dc_count}")


if __name__ == "__main__":
    main()
