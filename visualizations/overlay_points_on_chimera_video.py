import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def overlay_points_on_chimera_video(out_path, chimera_video_path, movement_series_list):
    """Assumes the chimera video is 1000x1000 and matches a 40x40 image size. This can be achieved using the "top
    view" chimera configuration script. Also assumes 1 fps at 100ns tip speed."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(out_path, fourcc, 2, (1000, 1000))

    vidcap = cv2.VideoCapture(chimera_video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(image.astype(np.uint8), 'RGB')
        image_draw = ImageDraw.Draw(im, "RGB")
        pixel_size = float(image.shape[0]) / 40.0
        for series in movement_series_list:
            if series.is_i_in_range(count):
                pos = series.get_ith_position(count - series.get_start_i())
                pos = (pos[0], 40 - pos[1])
                outline_coords = ((pos[0] - 0.3) * pixel_size, (pos[1] - 0.3) * pixel_size,
                                  (pos[0] + 1.3) * pixel_size, (pos[1] + 1.3) * pixel_size)
                image_draw.ellipse(outline_coords, fill=(0, 0, 0, 255))
                point_coords = ((pos[0] - 0.2) * pixel_size, (pos[1] - 0.2) * pixel_size,
                                (pos[0] + 1.2) * pixel_size, (pos[1] + 1.2) * pixel_size)
                image_draw.ellipse(point_coords, fill=(0, 0, 255, 255))
        video.write(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
        count += 1
        success, image = vidcap.read()

    video.release()
