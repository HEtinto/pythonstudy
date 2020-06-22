import numpy
import matplotlib
import glob
import imageio

# 测试神经网络是否能准确识别自己的手绘28*28 png图像


def text():
    # our own image test data set
    our_own_dataset = []

    # load the png image data as test data set
    for image_file_name in glob.glob('my_own_images/2828_my_own_?.png'):
        # use the filename to set the correct label
        label = int(image_file_name[-5:-4])

        # load image data from png files into an array
        print("loading ... ", image_file_name)
        img_array = imageio.imread(image_file_name, as_gray=True)

        # reshape from 28x28 to list of 784 values, invert values
        img_data = 255.0 - img_array.reshape(784)

        # then scale data to range from 0.01 to 1.0
        img_data = (img_data / 255.0 * 0.99) + 0.01
        print(numpy.min(img_data))
        print(numpy.max(img_data))

        # append label and image data  to test data set
        record = numpy.append(label, img_data)
        our_own_dataset.append(record)

        pass

    # test the neural network with our own images

    # record to test
    item = 2

    # plot image
    matplotlib.pyplot.imshow(our_own_dataset[item][1:].reshape(28, 28), cmap='Greys', interpolation='None')

    # correct answer is first value
    correct_label = our_own_dataset[item][0]
    # data is remaining values
    inputs = our_own_dataset[item][1:]

    # query the network
    outputs = numpy.query(inputs)
    print(outputs)

    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    print("network says ", label)
    # append correct or incorrect to list
    if (label == correct_label):
        print("Good,match!")
    else:
        print("no match!")
        pass
