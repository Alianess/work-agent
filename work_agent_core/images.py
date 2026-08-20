"""把图片变成模型真正用得上的样子。

一张手机原图 2.7MB / 4000×3000，base64 之后 3.7MB；两张就是 7MB，实测直接把
请求顶到 TPM 上限。而模型端最多用到长边 1568px——多出来的分辨率不提高任何
识别效果，只是账单。
"""

from __future__ import annotations

from pathlib import Path
import base64
import io


IMAGE_MAX_EDGE_PIXELS = 1024
"""实测出来的，不是猜的。

拿一页中文手写会议笔记做基准，人工读出 28 个锚点词，看模型转写命中几个：

    1024px  202KB  三次 [22, 21, 23]  均值 22.0
    1568px  408KB  三次 [22, 22, 24]  均值 22.7

差 0.7，而同一档跑三次的极差就有 2——**分辨率过了 1024px 买不到任何东西**。
模型报的 prompt_tokens 也印证这点：256→90、512→279、768→594、1024→1026、
1280→1014、1568→1014、2048→1014，过 1024 就不再增长，它内部自己降采样了。

另一头有硬下限：768px 以下模型不是读不出，是**会编**。同一张图在 512px 被
说成"辩论赛"、640px 说成"考研复试"、768px 说成"科技创新创业"——七个分辨率
七个答案，每个都说得很像。

所以取 1024：准确率和更高分辨率没有可测差别，字节数少 30%。而字节数才是成本
——限流按请求体算，一次会话就是被 7MB 的图撑到 429 全线失败的。
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
