from .camera import VideoCamera
from django.http import StreamingHttpResponse
from django.views.decorators.clickjacking import xframe_options_exempt
from django.http import HttpResponse


@xframe_options_exempt
def video_feed(request):
    # Get the camera index from the request parameters (default to 0 if not provided)
    camera_index = int(request.GET.get("camera_index", 0))

    # Create an instance of VideoCamera with the specified camera index
    video_camera = VideoCamera(camera_index=camera_index)

    response = StreamingHttpResponse(
        video_camera.generate_frames(),
        content_type="multipart/x-mixed-replace;boundary=frame",
    )

    # Set CORS headers explicitly

    return response
