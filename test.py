# test_one_hot = generate_one_hot(test_labels)

# test_batch_size = 52
# for i in range(len(test_labels)/test_batch_size):
#     test_spectro_batch = test_spectros[i*test_batch_size:(i*test_batch_size)+test_batch_size]
#     test_one_hot_batch = test_one_hot[i*test_batch_size:(i*test_batch_size)+test_batch_size]    
#     test_accuracy = model.accuracy.eval(feed_dict={model.x: test_spectro_batch, model.y_: test_one_hot_batch, model.keep_prob: 1.0})
#     print("Test accuracy %g" % test_accuracy)
