import os
from skimage import io
import tensorflow as tf
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

def save_images_from_event(fn, tag, output_dir='./'):
    """
    Code for saving tensorboard training images as png
    :param fn:
    :param tag:
    :param output_dir:
    :return:
    """
    assert(os.path.isdir(output_dir))

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf.train.summary_iterator(fn):
            for v in e.summary.value:
                if v.tag == tag:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    output_fn = os.path.realpath('{}/image_{:05d}.png'.format(output_dir, count))
                    print("Saving '{}'".format(output_fn))
                    io.imsave(output_fn, im.squeeze())

                    img = Image.open(output_fn)
                    draw = ImageDraw.Draw(img)
                    # font = ImageFont.truetype(<font-file>, <font-size>)
                    font = ImageFont.truetype("/usr/share/fonts/TTF/Roboto-Regular.ttf", 36)
                    # draw.text((x, y),"Sample Text",(r,g,b))
                    draw.text(xy=(620, 590), text="Iteration: {}".format(count*100), fill="#ffffff", font=font)
                    img.save(output_fn)


                    count += 1
save_images_from_event(fn="/home/edoardo/Projects/DoomPCGML/artifacts/tensorboard_results/reference_sample/events.out.tfevents.1521629976.446bf9710bed", tag="generator_output_rescaled/image/0", output_dir="/home/edoardo/Projects/DoomPCGML/artifacts/learningimages/")