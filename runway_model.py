# MIT License

# Copyright (c) 2019 Runway AI, Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import print_function

import torch
import process_stylization
from photo_wct import PhotoWCT

from PIL import Image

import runway
from runway.data_types import image, category


PRETRAINED_MODEL_PATH = './PhotoWCTModels/photo_wct.pth'


@runway.setup(options={'propagation_mode': category(description='Speeds up the propagation step by '
                                                                'using the guided image filtering algorithm',
                                                    choices=['fast', 'slow'],
                                                    default='fast')})
def setup(opts):
    p_wct = PhotoWCT()
    p_wct.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))

    if opts['propagation_mode'] == 'fast':
        from photo_gif import GIFSmoothing
        p_pro = GIFSmoothing(r=35, eps=0.001)
    else:
        from photo_smooth import Propagator
        p_pro = Propagator()
    if torch.cuda.is_available():
        p_wct.cuda(0)

    return {
        'p_wct': p_wct,
        'p_pro': p_pro,
    }


@runway.command(name='generate',
                inputs={ 'content': image(),
                         'style': image() },
                outputs={ 'image': image() })
def generate(model, args):
    p_wct = model['p_wct']
    p_pro = model['p_pro']

    # TODO: Use image directly instead of saving to path
    content_image_path = '/tmp/content.png'
    style_image_path = '/tmp/style.png'
    args['content'].save(content_image_path, 'PNG')
    args['style'].save(style_image_path, 'PNG')
    output_image_path = '/tmp/output.png'

    process_stylization.stylization(
        stylization_module=p_wct,
        smoothing_module=p_pro,
        content_image_path=content_image_path,
        style_image_path=style_image_path,

        # TODO: Allow passing in segmented images
        content_seg_path=[],
        style_seg_path=[],

        output_image_path=output_image_path,
        cuda=torch.cuda.is_available(),
        save_intermediate=False,
        no_post=False
    )

    return {
        # TODO: Pass PIL Image directly instead of loading from file
        'image': Image.open(output_image_path)
    }


if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=8888)
