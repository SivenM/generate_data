from transform import ImageGenerator


def main():
    images_dir_path = 'train_images/images_300'
    ann_path = 'train_images/labels.csv'
    new_img_dir_path = 'train_images/train_images_32'
    new_label_path = 'train_images/train_labels_32.csv'

    image_generator = ImageGenerator(images_dir_path, ann_path, new_img_dir_path, new_label_path)
    image_generator.generate_images()


if __name__ == "__main__":
    main()
