import cv2, os
import numpy as np
import matplotlib.image as mpimg
import sklearn

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(data_dir, image_file):
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def crop(image):
    return image[60:-25, :, :] # remove the sky and the car front


def resize(image):
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image

def random_flip(image, steering_angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def generator(samples, batch_size=30, dirimg='data'):
    num_samples = len(samples)
    print('num_samples', num_samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                correction = 0.1
                steering_angle = float(batch_sample[3])

                center_image = cv2.imread('./' + dirimg + '/IMG/' + batch_sample[0].split('/')[-1])
                center_image = preprocess(center_image)
                center_image, c_angle = random_flip(center_image, steering_angle)

                left_image = cv2.imread('./' + dirimg + '/IMG/' + batch_sample[1].split('/')[-1])
                left_image = preprocess(left_image)
                left_image, l_angle = random_flip(left_image, steering_angle)

                right_image = cv2.imread('./' + dirimg + '/IMG/' + batch_sample[2].split('/')[-1])
                right_image = preprocess(right_image)
                right_image, r_angle = random_flip(right_image, steering_angle)

                images.append(center_image)
                images.append(left_image)
                images.append(right_image)

                measurements.append(c_angle)
                measurements.append(l_angle + correction)
                measurements.append(r_angle - correction)

            x = np.array(images)
            y = np.array(measurements)
            yield sklearn.utils.shuffle(x, y)