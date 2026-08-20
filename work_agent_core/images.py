"""把图片变成模型真正用得上的样子。

一张手机原图 2.7MB / 4000×3000，base64 之后 3.7MB；两张就是 7MB，实测直接把
请求顶到 TPM 上限。而模型端最多用到长边 1568px——多出来的分辨率不提高任何
识别效果，只是账单。
"""

from __future__ import annotations

from pathlib import Path
import base64
import io


IMAGE_MAX_EDGE_PIXELS = 1280
"""实测出来的甜点，不是猜的。

同一张手写笔记按长边逐级上传，模型报的 prompt_tokens 是：
256px→90、512px→279、768px→594、1024px→1026、1280px→1014、1568px→1014。
**过了 1024px 就不再增长**——模型内部自己降采样了，多传的字节零收益却全额
计入限流配额。

另一头也有硬下限：768px 以下模型会编。同一张图在 512px 被说成"辩论赛"、
640px 被说成"考研复试"，到 1280px 才读出真实内容（租金、退出机制、知识产权）。
所以这个值只能落在 1024–1280 之间。
"""

IMAGE_JPEG_QUALITY = 80
IMAGE_MAX_ENCODED_BYTES = 400 * 1024
"""超过这个大小就重新编码。限流按请求体算，字节数才是真正的成本。"""


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

        from PIL import ImageOps

        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            # 手机照片的方向记在 EXIF 里，像素本身是躺着的。不摆正的话模型要
            # 先自己判断该转多少度——实测它会在回答里专门说明"已旋转 90 度"，
            # 那是白花的注意力。
            image = ImageOps.exif_transpose(image) or image
            if max(image.size) > IMAGE_MAX_EDGE_PIXELS or len(raw) > IMAGE_MAX_ENCODED_BYTES:
                image.thumbnail((IMAGE_MAX_EDGE_PIXELS, IMAGE_MAX_EDGE_PIXELS))
                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, format="JPEG", quality=IMAGE_JPEG_QUALITY, optimize=True)
                return "image/jpeg", base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        # 压不动就按原样发：宁可贵一点，也不要让附件消失。
        pass
    return mime_type, base64.b64encode(raw).decode("ascii")
