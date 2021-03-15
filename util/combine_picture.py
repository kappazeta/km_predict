import os
from PIL import Image
import numpy as np

# Take product name # Experiment path
# Output big original image, prediction, labels, sen2cor

# parse product name
# initialize big image
# loop through numbers
# PRODUCT = "S2A_MSIL2A_20200529T094041_N0214_R036_T35VLF_20200529T120441.CVAT"
# PRODUCT = "S2A_MSIL2A_20200509T094041_N0214_R036_T35VME_20200509T111504.CVAT"
PRODUCT = "S2B_MSIL2A_20200501T093029_N0214_R136_T35VMF_20200501T113501.CVAT"
EXPERIMENT_PATH = "/sar/data/cvat/code/cloudmask-fit/results/dice_loss_pr"
TILESIZE = 512

if __name__ == "__main__":
    # Parsing Product name
    file_specificator = PRODUCT.rsplit('.', 1)[0]
    date_match = file_specificator.rsplit('_', 1)[-1]
    index_match = file_specificator.rsplit('_', 1)[0].rsplit('_', 1)[-1]

    folder = EXPERIMENT_PATH + "/prediction"
    set_x = set()
    set_y = set()
    for subfolder in os.listdir(folder):
        if subfolder.startswith(index_match + "_" + date_match):
            splitting = subfolder.rsplit('_', 2)
            print(splitting)
            set_x.add(int(splitting[1]))
            set_y.add(int(splitting[-1]))

    big_image_x = (len(set_x) + 1) * TILESIZE
    big_image_y = (len(set_y) + 1) * TILESIZE

    # Generating 4 big images, 1 original, 1 prediction, 1 labels and 1 sen2cor
    big_orig = np.zeros((big_image_x, big_image_y, 3))
    big_pred = np.zeros((big_image_x, big_image_y))
    big_label = np.zeros((big_image_x, big_image_y))
    big_scl = np.zeros((big_image_x, big_image_y))

    for subfolder in os.listdir(folder):
        if subfolder.startswith(index_match + "_" + date_match):
            try:
                path = folder + "/" + subfolder
                pred_im = np.asarray(Image.open(folder + "/" + subfolder + "/prediction.png"))
                # TODO:: Label can be missing
                #label_im = np.asarray(Image.open(folder + "/" + subfolder + "/label.png"))
                orig_im = np.asarray(Image.open(folder + "/" + subfolder + "/orig.png"))
                scl_im = np.asarray(Image.open(folder + "/" + subfolder + "/SCL.png"))
                pred_im = np.flip(pred_im, 0)
                orig_im = np.flip(orig_im, 0)
                #label_im = np.flip(label_im, 0)
                scl_im = np.flip(scl_im, 0)
                splitting = subfolder.rsplit('_', 2)
                x = int(splitting[1])
                y = int(splitting[-1])
                big_orig[y * TILESIZE:(y + 1) * TILESIZE, x * TILESIZE:(x + 1) * TILESIZE, :] = orig_im
                big_pred[y * TILESIZE:(y + 1) * TILESIZE, x * TILESIZE:(x + 1) * TILESIZE] = pred_im
                #big_label[y * TILESIZE:(y + 1) * TILESIZE, x * TILESIZE:(x + 1) * TILESIZE] = label_im
                big_scl[y * TILESIZE:(y + 1) * TILESIZE, x * TILESIZE:(x + 1) * TILESIZE] = scl_im
            except:
                print("No subfolder " + subfolder)

    directory = EXPERIMENT_PATH + "/big_images"
    if not os.path.exists(directory):
        os.mkdir(directory)

    big_orig = big_orig.astype(np.uint8)
    im_orig = Image.fromarray(big_orig)
    im_orig.save(directory + "/" + index_match + "_" + date_match + "_im_orig.png")

    big_pred = big_pred.astype(np.uint8)
    im_pred = Image.fromarray(big_pred)
    im_pred.save(directory + "/" + index_match + "_" + date_match + "_im_pred.png")

    #big_label = big_label.astype(np.uint8)
    #im_label = Image.fromarray(big_label)
    #im_label.save(directory + "/" + index_match + "_" + date_match + "_im_label.png")

    big_scl = big_scl.astype(np.uint8)
    im_scl = Image.fromarray(big_scl)
    im_scl.save(directory + "/" + index_match + "_" + date_match + "_im_scl.png")
