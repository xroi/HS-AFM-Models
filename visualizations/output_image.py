from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mp
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable


def output_image(height_map, filename, min_z, max_z, res_x, res_y, colormap_name, add_legend=False,
                 crop_from_sides_px=0, draw_inner_circle_r=-1, draw_outer_circle_r=-1, add_scale=False):
    """z_center is the real center"""
    # Generate legend image
    if add_legend:
        legend_im = generate_legend_image(colormap_name, max_z, min_z, res_y)
        dims = (res_x + legend_im.size[0], res_y)
    else:
        dims = (res_x, res_y)
    if add_scale is True:
        scale_font = ImageFont.truetype('arial.ttf', 30)
    # Load the colormap
    cm = plt.get_cmap(colormap_name)

    if crop_from_sides_px > 0:
        height_map = height_map[crop_from_sides_px:-crop_from_sides_px, crop_from_sides_px:-crop_from_sides_px]
    im = generate_base_image(height_map, cm, max_z, min_z, res_x, res_y)
    image_draw = ImageDraw.Draw(im, "RGBA")
    pixel_size = res_x / (height_map.shape[0] - crop_from_sides_px * 2)
    if draw_inner_circle_r != -1:
        r = draw_inner_circle_r * pixel_size
        draw_circle(im, image_draw, r)
    if draw_outer_circle_r != -1:
        r = draw_outer_circle_r * pixel_size
        draw_circle(im, image_draw, r)
    if add_scale:
        add_scale_to_image(im, image_draw, pixel_size, scale_font)
    if add_legend:
        im = add_legend_to_image(im, legend_im)
    im.save(filename)


def draw_timestamp(image_draw, timestamp_font, timestamp_step, i):
    image_draw.text((30, 30), f"{(i * timestamp_step):.3f} μs", fill=(0, 0, 0, 255), font=timestamp_font)


def generate_base_image(height_map, cm, max_z, min_z, res_x, res_y):
    scaled_map = (height_map - min_z) / (max_z - 1 - min_z)
    data = cm(scaled_map)
    im = Image.fromarray((data[:, :, :3] * 255).astype(np.uint8), 'RGB')
    im = im.resize((res_y, res_x), resample=Image.BOX).rotate(angle=90, expand=True)
    return im


def add_legend_to_image(im, legend_im):
    new_im = Image.new('RGB', (im.size[0] + legend_im.size[0], im.size[1]), (250, 250, 250))
    new_im.paste(im, (0, 0))
    new_im.paste(legend_im, (im.size[0], 0))
    im = new_im
    return im


def add_scale_to_image(im, image_draw, pixel_size, scale_font):
    scale_text_coords = (im.size[0] - 7 * pixel_size, im.size[1] - 4.5 * pixel_size)
    scale_coords = [im.size[0] - 7 * pixel_size, im.size[1] - 3 * pixel_size,
                    im.size[0] - 2 * pixel_size, im.size[1] - 2 * pixel_size]
    image_draw.text(scale_text_coords, f"5 nm", fill=(0, 0, 0, 255), font=scale_font)
    image_draw.rectangle(scale_coords, fill="#000000")


def generate_legend_image(colormap_name, max_z, min_z, res_y):
    legend_fig = make_matplot_colorbar(0, max_z - min_z + 1, colormap_name)
    legend_im = fig2img(legend_fig)
    hpercent = (res_y / float(legend_im.size[1]))
    wsize = int((float(legend_im.size[0]) * float(hpercent)))
    legend_im = legend_im.resize((wsize, res_y), Image.Resampling.LANCZOS)
    legend_im = legend_im.crop((int((legend_im.size[0] / 2) + 80), 0, legend_im.size[0] - 250, legend_im.size[1]))
    return legend_im


def draw_circle(im, image_draw, r):
    image_draw.ellipse([(im.size[0] / 2 - r),
                        (im.size[1] / 2 - r),
                        (im.size[0] / 2 + r),
                        (im.size[1] / 2 + r)],
                       outline=(0, 0, 0, 125), width=5)


def make_matplot_colorbar(min, max, color_map):
    ax = plt.subplot()
    im = ax.imshow(np.arange(min, max, 5).reshape(int((max - min) / 5) + 1, 1), cmap=color_map)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="30%", pad=1)
    plt.colorbar(im, cax=cax, label="Height (nm)")
    # plt.show()
    return plt


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img
