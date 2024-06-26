from __future__ import absolute_import, division, print_function

import os
import sys

import PIL.Image as pil

import torch
from torchvision import transforms

sys.path.append("deps/monodepth")

import deps.monodepth.networks as networks
from deps.monodepth.utils import download_model_if_doesnt_exist

from seathru import *


def read_directory(directory_name):
    filenames = []
    for filename in os.listdir(directory_name):
        filenames.append(filename)
    return filenames


def run_folder(args):
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("On cuda")
    else:
        device = torch.device("cpu")
        print("On cpu")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("mono_models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    filenames = read_directory(args.input_folder)
    idx = 1
    size = len(filenames)

    for filename in filenames:
        try:
            input_path_name = args.input_folder + "\\" + filename
            output_path_name = args.output_folder + "\\" + filename
            if os.path.exists(output_path_name):
                print(f"{output_path_name} exist")
            else:
                img = Image.fromarray(rawpy.imread(input_path_name).postprocess()) if args.raw else pil.open(
                    input_path_name).convert('RGB')
                img.thumbnail((args.size, args.size))
                original_width, original_height = img.size
                # img = exposure.equalize_adapthist(np.array(img), clip_limit=0.03)
                # img = Image.fromarray((np.round(img * 255.0)).astype(np.uint8))
                input_image = img.resize((feed_width, feed_height), pil.LANCZOS)
                input_image = transforms.ToTensor()(input_image).unsqueeze(0)
                print('Preprocessed image', flush=True)

                # PREDICTION
                input_image = input_image.to(device)
                features = encoder(input_image)
                outputs = depth_decoder(features)

                disp = outputs[("disp", 0)]
                disp_resized = torch.nn.functional.interpolate(
                    disp, (original_height, original_width), mode="bilinear", align_corners=False)

                # Saving colormapped depth image
                disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
                mapped_im_depths = ((disp_resized_np - np.min(disp_resized_np)) / (
                        np.max(disp_resized_np) - np.min(disp_resized_np))).astype(np.float32)
                # print("Processed image", flush=True)
                # print('Loading image...', flush=True)
                depths = preprocess_monodepth_depth_map(mapped_im_depths, args.monodepth_add_depth,
                                                        args.monodepth_multiply_depth)
                recovered = run_pipeline(np.array(img) / 255.0, depths, args)
                # recovered = exposure.equalize_adapthist(scale(np.array(recovered)), clip_limit=0.03)
                sigma_est = estimate_sigma(recovered, average_sigmas=True, channel_axis=-1) / 10.0
                recovered = denoise_tv_chambolle(recovered, sigma_est)
                im = Image.fromarray((np.round(recovered * 255.0)).astype(np.uint8))
                im.save(output_path_name)
                print(f'Done. {output_path_name}', flush=True)
            print(f'{idx} / {size}', float(idx) / size, flush=True)
            idx += 1
        except Exception as e:
            print(e)


def run(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("On cuda")
    else:
        device = torch.device("cpu")
        print("On cpu")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("mono_models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # Load image and preprocess
    img = Image.fromarray(rawpy.imread(args.image).postprocess()) if args.raw else pil.open(args.image).convert('RGB')
    # img.thumbnail((args.size, args.size))
    original_width, original_height = img.size
    # img = exposure.equalize_adapthist(np.array(img), clip_limit=0.03)
    # img = Image.fromarray((np.round(img * 255.0)).astype(np.uint8))
    input_image = img.resize((feed_width, feed_height), pil.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    print('Preprocessed image', flush=True)

    # PREDICTION
    input_image = input_image.to(device)
    features = encoder(input_image)
    outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]
    disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width), mode="bilinear", align_corners=False)

    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
    mapped_im_depths = ((disp_resized_np - np.min(disp_resized_np)) / (
            np.max(disp_resized_np) - np.min(disp_resized_np))).astype(np.float32)
    print("Processed image", flush=True)
    print('Loading image...', flush=True)
    depths = preprocess_monodepth_depth_map(mapped_im_depths, args.monodepth_add_depth,
                                            args.monodepth_multiply_depth)
    recovered = run_pipeline(np.array(img) / 255.0, depths, args)
    # recovered = exposure.equalize_adapthist(scale(np.array(recovered)), clip_limit=0.03)
    sigma_est = estimate_sigma(recovered, average_sigmas=True, channel_axis=-1) / 10.0
    recovered = denoise_tv_chambolle(recovered, sigma_est)
    im = Image.fromarray((np.round(recovered * 255.0)).astype(np.uint8))
    im.save(args.output)
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', action='store_true')
    parser.add_argument('--input_folder', default='', help='input_folder')
    parser.add_argument('--output_folder', default='', help='output_folder')
    parser.add_argument('--image', default='input.png', help='Input image')
    parser.add_argument('--output', default='output.png', help='Output filename')
    parser.add_argument('--f', type=float, default=2.0, help='f value (controls brightness)')
    parser.add_argument('--l', type=float, default=0.5, help='l value (controls balance of attenuation constants)')
    parser.add_argument('--p', type=float, default=0.01, help='p value (controls locality of illuminant map)')
    parser.add_argument('--min-depth', type=float, default=0.0,
                        help='Minimum depth value to use in estimations (range 0-1)')
    parser.add_argument('--max-depth', type=float, default=1.0,
                        help='Replacement depth percentile value for invalid depths (range 0-1)')
    parser.add_argument('--spread-data-fraction', type=float, default=0.05,
                        help='Require data to be this fraction of depth range away from each other in attenuation estimations')
    parser.add_argument('--size', type=int, default=1920, help='Size to output')
    parser.add_argument('--monodepth-add-depth', type=float, default=2.0, help='Additive value for monodepth map')
    parser.add_argument('--monodepth-multiply-depth', type=float, default=10.0,
                        help='Multiplicative value for monodepth map')
    parser.add_argument('--model-name', type=str, default="mono_1024x320",
                        help='monodepth model name')
    parser.add_argument('--output-graphs', action='store_true', help='Output graphs')
    parser.add_argument('--raw', action='store_true', help='RAW image')

    args = parser.parse_args()
    if args.folder:
        run_folder(args)
    else:
        run(args)
