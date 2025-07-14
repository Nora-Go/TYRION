import os
import shutil
import cv2

if __name__ == '__main__':
    # for root_dir in ["data_raw", "data_raw_val_as_test_set"]:
    for root_dir in ["data_raw_val_as_test_set"]:
        data_raw_dir = os.path.join(os.path.dirname(os.getcwd()), root_dir)
        data_raw_dir_rot_0 = os.path.join(os.path.dirname(os.getcwd()), root_dir+"_rot_0")
        data_raw_dir_rot_1 = os.path.join(os.path.dirname(os.getcwd()), root_dir+"_rot_1")
        data_raw_dir_rot_2 = os.path.join(os.path.dirname(os.getcwd()), root_dir+"_rot_2")
        data_raw_dir_rot_3 = os.path.join(os.path.dirname(os.getcwd()), root_dir+"_rot_3")
        # os.makedirs(data_raw_dir_rot_0)
        # os.makedirs(data_raw_dir_rot_1)
        # os.makedirs(data_raw_dir_rot_2)
        # os.makedirs(data_raw_dir_rot_3)

        shutil.copytree(data_raw_dir, data_raw_dir_rot_0)
        shutil.rmtree(os.path.join(data_raw_dir_rot_0, "fronts", "train"))
        shutil.rmtree(os.path.join(data_raw_dir_rot_0, "sar_images", "train"))
        shutil.rmtree(os.path.join(data_raw_dir_rot_0, "zones", "train"))

        shutil.copytree(data_raw_dir_rot_0, data_raw_dir_rot_1)
        shutil.copytree(data_raw_dir_rot_0, data_raw_dir_rot_2)
        shutil.copytree(data_raw_dir_rot_0, data_raw_dir_rot_3)

        for rot, dir in [(1, data_raw_dir_rot_1), (2, data_raw_dir_rot_2), (3, data_raw_dir_rot_3)]:
            for root, dirs, files in os.walk(dir):
                for file in files:
                    if file.endswith(".png"):
                        # read in the image and rotate it by rot*90 degrees
                        img = cv2.imread(os.path.join(root, file))
                        for i in range(rot):
                            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                        cv2.imwrite(os.path.join(root, file), img)
