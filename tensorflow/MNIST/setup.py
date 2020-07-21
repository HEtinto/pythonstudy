# 在不同的环境中引用库的方式可能不同
# from MNIST.mnsit import *
# from MNIST.plot import *

# 用于在控制台中运行
from mnsit import *
from plot import *

for epoch in range(train_epochs):
    for batch in range(total_batch):
        xs, ys = mnist.train.next_batch(batch_size)  # 读取批次数据
        sess.run(optimizer, feed_dict={x:xs, y:ys})  # 执行批次训练
    # total_batch个批次训练完成后，使用验证数据计算准确率；验证集没有分批
    loss, acc = sess.run([loss_function, accuracy], feed_dict=
    {x:mnist.validation.images, y:mnist.validation.labels})
    # 打印训练过程的详细信息
    if (epoch + 1) % display_step ==0:
        print("Train Epoch:", '%02d' % (epoch + 1),
              "Loss=", "{:.9f}".format(loss), "Accuracy=",
              "{:.4f}".format(acc))
        print("Train Finshed!")


accu_test = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
print("Test Accuracy", accu_test)


if __name__ == '__main__':
    prediction_result = sess.run(tf.argmax(pred, 1), feed_dict={x:mnist.test.images})
    print(prediction_result[0:10])
    plot_images_labels_prediction(mnist.test.images, mnist.test.labels, prediction_result, 0,
                                  15)  # 0代表下标从0幅开始，15表示最多显示15幅