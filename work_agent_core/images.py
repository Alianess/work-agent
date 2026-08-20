"""把图片变成模型真正用得上的样子。

一张手机原图 2.7MB / 4000×3000，base64 之后 3.7MB；两张就是 7MB，实测直接把
请求顶到 TPM 上限。而模型端最多用到长边 1568px——多出来的分辨率不提高任何
识别效果，只是账单。
"""

from __future__ import annotations

from pathlib import Path
import base64
import io


IMAGE_MAX_EDGE_PIXELS = 1568
"""模型端用不到更高的分辨率，多出来的像素只是账单。"""

IMAGE_MAX_ENCODED_BYTES = 900 * 1024


def encode_image_for_model(path: Path, mime_type: str) -> tuple[str, str]:
    """把图片压到模型真正用得上的尺寸，再编码。

    一张 iPhone 原图 2.7MB / 4000×3000，base64 之后 3.7MB；两张就是 7MB。
    而模型最多用到长边 1568px——多出来的分辨率一点用没有，纯烧 token，
    实测直接把请求顶到 TPM 上限。
    """

    try:
        raw = path.read_bytes()
    except OSError:
        return "", ""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            if max(image.size) > IMAGE_MAX_EDGE_PIXELS or len(raw) > IMAGE_MAX_ENCODED_BYTES:
                image.thumbnail((IMAGE_MAX_EDGE_PIXELS, IMAGE_MAX_EDGE_PIXELS))
                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, format="JPEG", quality=85, optimize=True)
                return "image/jpeg", base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        # 压不动就按原样发：宁可贵一点，也不要让附件消失。
        pass
    return mime_type, base64.b64encode(raw).decode("ascii")
