from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from argparse import ArgumentParser
from model.utils import *
from flopth import flopth


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--model_config', default="Tyrion.yaml", help="Name of the model config file for testing")

    hparams = parser.parse_args()
    model_config = hparams.model_config
    data_config = "test.yaml"

    model = instantiate_completely('model', model_config)
    model.eval()
    model.cuda()
    flops, params = flopth(model, in_size=((1, 512, 512),))
    print(f"FLOPS: {flops}, Params: {params}")
