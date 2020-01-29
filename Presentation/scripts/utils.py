import geopandas as gpd
import pandas as pd
from datetime import datetime
import glob
from tensorflow.python.framework import ops


def data_importer(path, all=False, label=True, multi_index=False) -> gpd.GeoDataFrame:
	csv_file_path = path

	with open(csv_file_path, 'r') as f:
		print('Importing data...')
		df = pd.read_csv(f)

		if not all and label:
			df = df.loc[list(map(lambda x: isinstance(x, str), df['mode']))]
		elif not all and not label:
			df = df.loc[list(map(lambda x: not isinstance(x, str), df['mode']))]
	
	gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

	print('Import complete! Wait a little moment for some other settings...')

	# change user id from float to str
	gdf['user_id'] = list(map(lambda x: '0' * (3 - len(str(int(x)))) + str(int(x)), gdf['user_id']))

	# change datetime from str to datetime.datetime
	gdf['datetime'] = list(map(lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'), gdf['datetime'].values))

	# sort all values according to user ID and datetime
	gdf = gdf.sort_values(by=['user_id', 'datetime'])
	gdf.index = [i for i in range(len(gdf))]

	# multi index
	if multi_index:
		gdf.index = pd.MultiIndex.from_arrays([gdf['user_id'], gdf['datetime']], names=['user_id', 'datetime'])

	print('All complete!')
	return gdf


def convert_to_one_hot(Y, num_features):
    Y = np.eye(num_features)[Y.reshape(-1)].T
    return Y


def calculate_box_plot_characteristics(my_list: list):
    result = {}
    
    result["minimum"] = np.min(my_list)
    result["maximum"] = np.max(my_list)
    result["median"] = np.median(my_list)
    
    q1 = np.percentile(my_list, 25)
    q3 = np.percentile(my_list, 75)
    iqr = q3 - q1
    result["lower_quartile"] = q1
    result["upper_quartile"] = q3
    
    lower_whisker = q1 - 1.5 * iqr
    upper_whisker = q3 + 1.5 * iqr
    rpa_sort = np.sort(my_list)
    for i in range(len(rpa_sort)):
        if rpa_sort[i] > lower_whisker:
            result["lower_whisker"] = rpa_sort[i]
            break
    for i in reversed(range(len(rpa_sort))):
        if rpa_sort[i] < upper_whisker:
            result["upper_whisker"] = rpa_sort[i]
            break
    
    return result


def report_hyperparams_and_results(d):
	with open('./NN_log.txt', 'a') as f:
		for key in d.keys():
			f.write(str(key) + ': ' + str(d[key]) + '\n')
		f.write('-----------------------------------\n')

	print('Write complete!')


#####################################################
# NN related functions
def initialize_parameters(params_size: list):
    """
    Initializes parameters to build a neural network with tensorflow.

    
    Returns:
    parameters -- a dictionary of tensors containing list W and list b
    """

    W = [None for i in range(len(params_size) - 1)]
    b = [None for i in range(len(params_size) - 1)]

    for i in range(len(params_size) - 1):
      W[i] = tf.get_variable('W'+str(i), [params_size[i+1], params_size[i]], initializer = tf.contrib.layers.xavier_initializer())
      b[i] = tf.get_variable('b'+str(i), [params_size[i+1], 1], initializer = tf.zeros_initializer())

    parameters = {'W': W, 'b': b}
    
    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: (LINEAR -> RELU) x k -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing parameters list W and list b

    Returns:
    Z_out -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W = parameters['W']
    b = parameters['b']

    Z = [None for i in range(len(W) - 1)]
    A = [None for i in range(len(W) - 1)]

    for i in range(len(W)):
      if i == 0:
        Z[i] = tf.matmul(W[i], X) + b[i]
        A[i] = tf.nn.relu(Z[i])
      elif i != len(W) - 1:
        Z[i] = tf.matmul(W[i], A[i-1]) + b[i]
        A[i] = tf.nn.relu(Z[i])
      else:
        Z_out = tf.matmul(W[i], A[i-1]) + b[i]

    return Z_out


def compute_cost(Z_out, Y, parameters, lambd, m):
    """
    Compute cost using L2 regularization.
    
    Arguments:
    Z_out -- the output of the last LINEAR unit
    Y -- labels placeholder, of shape (number of classes, number of examples)
    parameters -- python dictionary containing parameters list W and list b
    lambd -- L2 Regularization parameter
    m -- number of examples used to calculate L2 Regularization

    Returns:
    cost -- cost with L2 regularization
    """
    # Retrieve the parameters from the dictionary "parameters" 
    W = parameters['W']
    
    logits = tf.transpose(Z_out)
    labels = tf.transpose(Y)
    
    cost_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    cost_L2_regularization = 1/m * lambd/2 * np.sum([tf.square(tf.norm(W[i], ord=2)) for i in range(len(W))])
    # cost_L2_regularization = 1/m * lambd/2 * (tf.square(tf.norm(W1, ord=2)) + tf.square(tf.norm(W2, ord=2)) + tf.square(tf.norm(W3, ord=2)) + tf.square(tf.norm(W4, ord=2)))
    # cost = cost_cross_entropy + cost_L2_regularization
    cost = cost_cross_entropy
    
    return cost


def random_mini_batches(X, Y, mini_batch_size):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector
    mini_batch_size - size of the mini-batches, integer
    seed -- random seed
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    # np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def model(X_train, Y_train, X_test, Y_test, params_size, learning_rate = 0.005,
          num_epochs = 100, minibatch_size = 1024, lambd = 0.2, print_cost = True):
    """
    Implements a neural network: (LINEAR -> RELU) x k -> LINEAR -> SOFTMAX.
    
    Arguments:
    X_train -- training set
    Y_train -- test set
    X_test -- training set
    Y_test -- test set
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    lambd -- L2 Regularization parameter
    print_cost -- True to print the cost every epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    X = tf.placeholder(tf.float32, [n_x, None], name='X')
    Y = tf.placeholder(tf.float32, [n_y, None], name='Y')

    # Initialize parameters
    parameters = initialize_parameters(params_size)
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z_out = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z_out, Y, parameters, lambd, minibatch_size)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                correct_prediction = tf.equal(tf.argmax(Z_out), tf.argmax(Y))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
                print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
                print("------------")
            if print_cost == True:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z_out), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters
