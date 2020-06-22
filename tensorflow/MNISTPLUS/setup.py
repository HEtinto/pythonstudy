from MNISTPLUS.classNetwork import *
# from MNISTPLUS.plot import *
# from plot import *
# from setup import *
# number of input,hidden and output nodes
# 28 * 28 = 784
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate is 0.3
learning_rate = 0.1

# create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# train the neural network

# load the mnist training data csv file into a list
training_data_file = open("MNIST_data/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# epochs is the number of times the training data set is used for training
epochs = 5
for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass


# test the neural network

# load the mnist test data csv file to a list
test_data_file = open("MNIST_data/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# scorecard for how well the network performs,initially empty
scorecard = []
# go through all records in the test data set
for record in test_data_list:
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    #    print("Answer label is:",correct_label," ; ",label," is network's answer")
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        scorecard.append(0)
    pass

# calculate the performance score ,the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)

# 测试神经网络是否能准确识别自己的手绘28*28 png图像

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
outputs = n.query(inputs)
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
