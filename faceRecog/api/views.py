from .camera import VideoCamera
from django.http import StreamingHttpResponse
from django.views.decorators.clickjacking import xframe_options_exempt
from django.http import HttpResponse


@xframe_options_exempt
def video_feed(request):
    response = StreamingHttpResponse(
        VideoCamera().generate_frames(),
        content_type="multipart/x-mixed-replace;boundary=frame",
    )

    # Set CORS headers explicitly

    return response
