from django.http import StreamingHttpResponse
from .camera import VideoCamera


def video_feed(request):
    return StreamingHttpResponse(
        VideoCamera().generate_frames(),
        content_type="multipart/x-mixed-replace;boundary=frame",
    )
