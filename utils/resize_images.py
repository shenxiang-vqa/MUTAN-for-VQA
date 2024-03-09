import os
import sys
import cv2
from tqdm import tqdm
import argparse

def resize_image(image,size):
    r"This fuction does that by resizing a image"
    return cv2.resize(image,size,interpolation=cv2.INTER_AREA)
def resize_images(input_dir,output_dir,size):
    """
    :param input_dir: 原始图像的路径 /data/lzm/dataset/VisionQA/VQAv2/images
    :param output_dir: 输出resize图像的路径 /data/lzm/dataset/VisionQA/VQAv2/resize_images
    :return:
    """
    for idir in os.scandir(input_dir):
        """
        A example is as follows:
        images/
            test2015/
                1.jpg
                2.jpg
            train2014/
                1.jpg
                2.jpg
            val2014/
                1.jpg
                2.jpg
        with os.scandir(input_dir) as entries:
            for entry in entries:
                print(entry.name)            # 文件或目录的名称
                out:test2015
                print(entry.path)            # 完整路径
                out:images/test2015
                print(entry.is_file())       # 是否为文件
                out:False
                print(entry.is_dir()) #是否为文件夹
                out:True
        """

        if not idir.is_dir():
            continue
        if not os.path.exists(output_dir+'/'+idir.name):#If the path does not exist
            os.makedirs(output_dir+'/'+idir.name)


        images = os.listdir(idir.path)
        """
        os.listdir(test2015) return ['1.jpg', '2.jpg']
        """
        n_images = len(images)

        for idx_image , image in tqdm(enumerate(images)): #tqdm return 2 values : first value is index , second vales is image
            f = os.path.join(idir.path,image) #os.path.join(path1,path2) is connect path1 and path2 as new path
            img = cv2.imread(f) # open an image in the f-path
            img = resize_image(img,size) #resize the image
            cv2.imwrite(os.path.join(output_dir+'/'+idir.name , image) , img) #Save new image to Specify path

            if (idx_image+1) % 1000 == 0: # print an information for every 1000 images processed
                print("[{} / {}] resize image and save image ing...".format(idx_image+1,n_images))

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    image_size = (args.image_size,args.image_size)
    resize_images(input_dir,output_dir,image_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='D:/data/VQA_data/VQAv2/images',
                        help='directory for input images (unresized images)')

    parser.add_argument('--output_dir', type=str, default='D:/data/VQA_data/VQAv2/resized448_images',
                        help='directory for output images (resized images)')

    parser.add_argument('--image_size', type=int, default=448,
                        help='size of images after resizing')

    args = parser.parse_args()

    main(args)

