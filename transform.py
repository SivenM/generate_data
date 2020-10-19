import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pylab as plt


class ImageTransformer:

    """"
    Генерирует координаты  новых изображений и  их аннотацию
    """

    def __init__(self, bbox):
        """

        :param bbox: аннотация исходного изображения
        """
        self.bbox = bbox

    def generate_buffers_40(self, image_number):
        """
        Генерирует  5 изображений. В зависимости от значения image_number объект и его аннотация генерируются в
        определенном месте будущего изображения.

        :param image_number: целочисленное значение, характеризует номер изображения
        :return:
            bbox: координаты генерируемого изображения
            ann_box: аннотация объекта в генерируемом изображении
        """
        coef = 40
        bbox = self.bbox.copy()
        ann_box = []
        if image_number == 0:
            bbox[1] = bbox[1] - coef
            bbox[2] = bbox[2] + coef
            bbox[bbox < 0] = 0
            ann_box = [0, coef, abs(bbox[2] - bbox[0]) - coef, abs(bbox[3] - bbox[1])]
        elif image_number == 1:
            bbox[0] = bbox[0] - coef
            bbox[1] = bbox[1] - coef
            bbox[bbox < 0] = 0
            ann_box = [coef, coef, abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1])]
        elif image_number == 2:
            bbox[0] = bbox[0] - coef
            bbox[3] = bbox[3] + coef
            bbox[bbox < 0] = 0
            ann_box = [coef, 0, abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1]) - coef]
        elif image_number == 3:
            bbox[2] = bbox[2] + coef
            bbox[3] = bbox[3] + coef
            bbox[bbox < 0] = 0
            ann_box = [0, 0, abs(bbox[2] - bbox[0]) - coef, abs(bbox[3] - bbox[1]) - coef]
        elif image_number == 4:
            bbox[0] = bbox[0] - (coef / 2)
            bbox[1] = bbox[1] - (coef / 2)
            bbox[3] = bbox[3] + (coef / 2)
            bbox[2] = bbox[2] + (coef / 2)
            bbox[bbox < 0] = 0
            ann_box = [coef / 2, coef / 2, abs(bbox[2] - bbox[0]) - (coef / 2), abs(bbox[3] - bbox[1]) - (coef / 2)]
        return bbox, ann_box

    def generate_buffers_30(self, image_number):
        """
            Генерирует  3 изображений. В зависимости от значения image_number объект и его аннотация генерируются в
            определенном месте будущего изображения.
            :param image_number: целочисленное значение, характеризует номер изображения
            :return:
                bbox: координаты генерируемого изображения
                ann_box: аннотация объекта в генерируемом изображении
        """
        bbox = self.bbox.copy()
        coef = 30
        ann_box = []
        if image_number == 0:
            bbox[1] = bbox[1] - (coef / 2)
            bbox[2] = bbox[2] + coef
            bbox[3] = bbox[3] + (coef / 2)
            bbox[bbox < 0] = 0
            ann_box = [0, (coef / 2), abs(bbox[2] - bbox[0]) - coef, abs(bbox[3] - bbox[1]) - (coef / 2)]
        elif image_number == 1:
            bbox[1] = bbox[1] - (coef / 2)
            bbox[0] = bbox[0] - coef
            bbox[3] = bbox[3] + (coef / 2)
            bbox[bbox < 0] = 0
            ann_box = [coef, (coef / 2), abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1]) - (coef / 2)]
        elif image_number == 2:
            bbox[0] = bbox[0] - (coef / 2)
            bbox[1] = bbox[1] - (coef / 2)
            bbox[3] = bbox[3] + (coef / 2)
            bbox[2] = bbox[2] + (coef / 2)
            bbox[bbox < 0] = 0
            ann_box = [(coef / 2), (coef / 2), abs(bbox[2] - bbox[0]) - (coef / 2), abs(bbox[3] - bbox[1]) - (coef / 2)]
        return bbox, ann_box

    def generate_buffers_10(self, image_number):
        """
            Генерирует  изображение. В зависимости от значения image_number объект и его аннотация генерируются в
            определенном месте будущего изображения.
            :param image_number: целочисленное значение, характеризует номер изображения
            :return:
                bbox: координаты генерируемого изображения
                ann_box: аннотация объекта в генерируемом изображении
        """
        bbox = self.bbox.copy()
        coef = 5
        ann_box = []
        bbox[0] = bbox[0] - coef
        bbox[1] = bbox[1] - coef
        bbox[3] = bbox[3] + coef
        bbox[2] = bbox[2] + coef
        bbox[bbox < 0] = 0
        ann_box = [coef, coef, abs(bbox[2] - bbox[0]) - coef, abs(bbox[3] - bbox[1]) - coef]
        return bbox, ann_box


class ImageGenerator:
    """
    Создает обучающий датасет для ssd_32. Трансформер выполяет слудющие функции:
    1. Создает 9 изображений из 1 аннотации объекта.
    2. Проверяет, если IOU > 0.2 между полученным изображением и другими аннотациями, то  полученное изображение
       удаляется.
    3. Ресайзит и записывает полученные изображения в формате .jpg
    4. Ресайзит и записывает полученные аннотаций csv-файл
    """

    def __init__(self, img_dir_path, ann_path, new_img_dir_path, csv_name='test_csv.csv', output_size=32):
        """
        :param img_dir_path: путь до  директории изображений
        :param ann_path: путь до аннотации
        :param new_img_dir_path: путь до директории, куда будут сохраняться новые изображения
        :param csv_name: имя csv-файла, куда будет записываться аннотация для полученных изображений
        :param output_size: размер генерируемых изображений
        """
        self.csv_name = csv_name
        self.img_dir_path = img_dir_path
        self.new_dir_path = new_img_dir_path
        self.img_names = self._get_img_names(img_dir_path)
        self.ann_df = self._get_ann_df(ann_path)
        self.new_df = pd.DataFrame(columns=self.ann_df.columns)
        self.output_size = output_size
        self.max_images = 9
        self.buffers = [40, 30, 10]

    @staticmethod
    def _get_img_names(path):
        """
        Возвращяет названия изображений, которые хранятся в path
        :param path: путь до изображений
        :return: list с названиями изображений
        """
        return os.listdir(path)

    @staticmethod
    def _get_ann_df(path):
        """
        Создает Dataframe из csv-файла
        :param path: путь где хранится csv-файл
        :return: Dataframe, который хранит аннотацию
        """
        return pd.read_csv(path)

    def _get_bbox_list(self, img_name):
        """
        Ищет аннотации азображения по  его имени
        :param img_name: имя изображения
        :return: список аннотаций на изображение
        """
        bbox_list = []
        index_list = []
        for i, name in enumerate(self.ann_df['filename']):
            if name == img_name:
                bbox_coords_df = self.ann_df.loc[i][4:]
                bbox_list.append(np.array([bbox_coords for bbox_coords in bbox_coords_df]))

        return np.stack(bbox_list)

    def compute_iou(self, boxes, buffer):
        """
        Считает Intersection Over Union между координатами аннотаций и генерируемым изображением
        :param boxes: Матрица размером [число аннотаций на изображение, 4]
        :param buffer: Матрица размером [1,4]
        :return: матрица размером [число аннотаций на изображение, 4], которая хранит IOU для каждой аннотации
        """
        lu = np.maximum(boxes[:, :2], buffer[:2])
        rd = np.minimum(boxes[:, 2:], buffer[2:])
        intersection = np.maximum(0.0, rd - lu)
        intersection_area = intersection[:, 0] * intersection[:, 1]
        box1_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        # box2_area = (buffer[2] - buffer[0]) * (buffer[3] - buffer[1])
        # union_area = np.maximum(box1_area + box2_area - intersection_area, 1e-8)
        return intersection_area / box1_area

    def check_iou(self, other_boxes, buffer_image):
        """
        Если IOU > 0.2 то изображения не генерируется
        :param other_boxes: массив размером [число аннотаций на изображение - 1, 4]
        :param buffer_image: Матрица размером [1,4] координаты генерируемого изображения
        :return: булево значение
        """
        iou = self.compute_iou(other_boxes, buffer_image)
        if len(iou[iou > 0.2]):
            return False
        else:
            return True

    def _crop(self, image_list, main_image):
        """
        Кропает генерируемые изображения из изначального
        :param image_list: список координат генерируемых изображений
        :param main_image: изначальное изображение
        :return: список генерируемых изображений
        """
        crop_images = []
        for image_box in image_list:
            crop_image = main_image[image_box[1]: image_box[3], image_box[0]: image_box[2]]
            crop_images.append(crop_image)
        return crop_images

    def resize(self, crop_images, ann_list):
        """
        Изменяет размер изображения и аннотации
        :param crop_images: список изображений
        :param ann_list: список аннотаций
        :return: списки изображений и аннотаций с новыми размерами
        """
        resized_images = []
        resized_ans = []
        for i, image in enumerate(crop_images):
            ann = ann_list[i]
            coef_x = image.shape[1] / self.output_size
            coef_y = image.shape[0] / self.output_size
            resized_image = cv2.resize(image, (self.output_size, self.output_size))
            resized_ann = [int(ann[0] / coef_x), int(ann[1] / coef_y), int(ann[2] / coef_x), int(ann[3] / coef_y)]
            resized_images.append(resized_image)
            resized_ans.append(resized_ann)
        return resized_images, resized_ans

    def _crop_and_resize_bbox(self, buffer, main_image, bbox, other_boxes):
        """
        1. Создает новые изображения из аннотации, если IOU < 0.2
        с другими аннотациями на изображении.
        2. Ресайзит полученные изображения и аннотации

        :param buffer: целочисленное значение, которое добавляется к целевому боксу для генерирования изображения
        :param main_image: исходное изображение
        :param bbox: целевой бокс
        :param other_boxes: остальные боксы в изображении
        :return: списки изображений и аннотаций с новыми размерами
        """
        transformer = ImageTransformer(bbox)
        max_images = 1
        generate_buffers = transformer.generate_buffers_10
        image_list = []
        ann_list = []

        if buffer == 40:
            max_images = 5
            generate_buffers = transformer.generate_buffers_40
        elif buffer == 30:
            max_images = 3
            generate_buffers = transformer.generate_buffers_30

        for i in range(max_images):
            buffer_image, ann_box = generate_buffers(i)
            if len(other_boxes) != 0:
                # Если людей больше чем 1
                if self.check_iou(other_boxes, buffer_image):
                    image_list.append(buffer_image)
                    ann_list.append(ann_box)
            else:
                image_list.append(buffer_image)
                ann_list.append(ann_box)
        crop_images = self._crop(image_list, main_image)
        resized_images, resized_anns = self.resize(crop_images, ann_list)
        return resized_images, resized_anns

    def _save_image_and_ann(self, resized_images, resized_anns, image_name, count_image, ann_count):
        """
        Сохраняет полученные изображения в указанную директорию, а  аннотации в DataFrame
        :param resized_images: сгенерированные изображения
        :param resized_anns: аннотация сгенерированных изображений
        :param image_name: имя исходного изображения
        :param count_image: счетчик изображений
        :param ann_count: счетчик аннотаций
        :return: счетчики для правльной группировки данных
        """
        for buffer_value in resized_images.keys():
            for i, image in enumerate(resized_images[buffer_value]):
                ann = resized_anns[buffer_value][i]
                resized_image_name = image_name[:-4] + '_' + str(count_image) + '_' + buffer_value + '_' + str(i) + '.jpg'
                resized_full_image_name = self.new_dir_path + '/' + resized_image_name
                cv2.imwrite(resized_full_image_name, image)

                print(f'{resized_full_image_name}  сохранен')
                print('=' * 60)
                self.new_df.loc[ann_count] = [resized_image_name,
                                              self.output_size,
                                              self.output_size,
                                              'person',
                                              ann[0],
                                              ann[1],
                                              ann[2],
                                              ann[3]
                                              ]
                ann_count += 1
        count_image += 1
        return ann_count, count_image

    def generate_images(self):
        """
        Основной метод, который генерирует изображения и аннотации, также сохраняет их.
        """
        ann_count = 0
        for image_name in self.img_names:
            print(image_name)
            main_image = cv2.imread(self.img_dir_path + '/' + image_name)
            label_boxes = self._get_bbox_list(image_name)
            num_boxes = label_boxes.shape[0]
            count_image = 0
            for i in range(num_boxes):
                bbox = label_boxes[i]
                other_boxes = np.delete(label_boxes.copy(), i, axis=0)
                resized_images = {}
                resized_anns = {}
                for buffer in self.buffers:
                    resized_images[str(buffer)], resized_anns[str(buffer)] = self._crop_and_resize_bbox(buffer,
                                                                                                        main_image,
                                                                                                        bbox,
                                                                                                        other_boxes)

                ann_count, count_image = self._save_image_and_ann(resized_images, resized_anns, image_name, count_image,
                                                                  ann_count)
        self.new_df.to_csv(self.csv_name, index=False)
        print('Данные успешно сгенерированы!')
