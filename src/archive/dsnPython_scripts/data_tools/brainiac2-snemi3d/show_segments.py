import argparse
import skimage.io as skio
import matplotlib.pyplot as plt
from dsn.vis_tools import overlay_img_with_labels

def parse_arguments():
    """
    Parses arguments and returns a dictionary having
    argument values.
    """
    p = argparse.ArgumentParser(description='''Plots an overlay of image and
    its corresponding labels.''')

    p.add_argument('img_fpath', type=str,
                   help=''' Full path of image''')
    p.add_argument('labels_fpath', type=str, help=''' full path of file having
    segmentation labels.''')

    # Build dictionary
    args                      = p.parse_args()
    args_dict                 = {}
    args_dict['img_fpath']    = args.img_fpath
    args_dict['labels_fpath'] = args.labels_fpath

    return args_dict
                   
    
    
if __name__ == '__main__':
    args_dict  = parse_arguments()

    # Read image and labels
    ori_img    = skio.imread(args_dict['img_fpath'])
    lab_img    = skio.imread(args_dict['labels_fpath'])

    # Create an image with labels overlayed
    ovr_img    = overlay_img_with_labels(ori_img, lab_img)
    
    # Overlay labels on original image and plot using matplotlib
    fig, ax = plt.subplots()
    
    
    plt.show()


    
    
    
