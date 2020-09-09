import numpy as np

from model_utils import softmax, rnn_forward, rnn_backward, update_parameters, initialize_parameters, get_initial_loss, \
    smooth, print_sample


class ModelService:

    def __init__(self):
        pass

    ### GRADED FUNCTION: clip
    @staticmethod
    def clip(gradients, maxValue):
        '''
        Clips the gradients' values between minimum and maximum.

        Arguments:
        gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
        maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue

        Returns:
        gradients -- a dictionary with the clipped gradients.
        '''

        dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients[
            'dby']

        ### START CODE HERE ###
        # clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]. (≈2 lines)
        min_value = -maxValue
        for gradient in [dWax, dWaa, dWya, db, dby]:
            np.clip(gradient, a_min=min_value, a_max=maxValue, out=gradient)
        ### END CODE HERE ###

        gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

        return gradients

    # GRADED FUNCTION: sample
    @staticmethod
    def sample(parameters, char_to_ix, seed):
        """
        Sample a sequence of characters according to a sequence of probability distributions output of the RNN

        Arguments:
        parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b.
        char_to_ix -- python dictionary mapping each character to an index.
        seed -- used for grading purposes. Do not worry about it.

        Returns:
        indices -- a list of length n containing the indices of the sampled characters.
        """

        # Retrieve parameters and relevant shapes from "parameters" dictionary
        Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters[
            'b']
        vocab_size = by.shape[0]
        n_a = Waa.shape[1]

        ### START CODE HERE ###
        # Step 1: Create the a zero vector x that can be used as the one-hot vector
        # representing the first character (initializing the sequence generation). (≈1 line)
        x = np.zeros(shape=(vocab_size, 1))
        # Step 1': Initialize a_prev as zeros (≈1 line)
        a_prev = np.zeros(shape=(n_a, 1))

        # Create an empty list of indices, this is the list which will contain the list of indices of the characters to generate (≈1 line)
        indices = []

        # idx is the index of the one-hot vector x that is set to 1
        # All other positions in x are zero.
        # We will initialize idx to -1
        idx = -1

        # Loop over time-steps t. At each time-step:
        # sample a character from a probability distribution
        # and append its index (`idx`) to the list "indices".
        # We'll stop if we reach 50 characters
        # (which should be very unlikely with a well trained model).
        # Setting the maximum number of characters helps with debugging and prevents infinite loops.
        counter = 0
        newline_character = char_to_ix['\n']

        while (idx != newline_character and counter != 50):
            # Step 2: Forward propagate x using the equations (1), (2) and (3)
            a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
            z = np.dot(Wya, a) + by
            y = softmax(z)

            # for grading purposes
            np.random.seed(counter + seed)

            # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
            # (see additional hints above)
            idx = np.random.choice(vocab_size, 1, p=y.ravel())[0]

            # Append the index to "indices"
            indices.append(idx)

            # Step 4: Overwrite the input x with one that corresponds to the sampled index `idx`.
            # (see additional hints above)
            x = np.zeros(shape=(vocab_size, 1))
            x[idx] = 1

            # Update "a_prev" to be "a"
            a_prev = a

            # for grading purposes
            seed += 1
            counter += 1

        if (counter == 50):
            indices.append(char_to_ix['\n'])

        return indices

    # GRADED FUNCTION: optimize
    @staticmethod
    def optimize(X, Y, a_prev, parameters, learning_rate=0.01):
        """
        Execute one step of the optimization to train the model.

        Arguments:
        X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
        Y -- list of integers, exactly the same as X but shifted one index to the left.
        a_prev -- previous hidden state.
        parameters -- python dictionary containing:
                            Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                            Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                            Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                            b --  Bias, numpy array of shape (n_a, 1)
                            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
        learning_rate -- learning rate for the model.

        Returns:
        loss -- value of the loss function (cross-entropy)
        gradients -- python dictionary containing:
                            dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                            dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                            dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                            db -- Gradients of bias vector, of shape (n_a, 1)
                            dby -- Gradients of output bias vector, of shape (n_y, 1)
        a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
        """

        ### START CODE HERE ###

        # Forward propagate through time (≈1 line)
        loss, cache = rnn_forward(X, Y, a_prev, parameters)

        # Backpropagate through time (≈1 line)
        gradients, a = rnn_backward(X, Y, parameters, cache)

        # Clip your gradients between -5 (min) and 5 (max) (≈1 line)
        gradients = ModelService.clip(gradients, 5)

        # Update parameters (≈1 line)
        parameters = update_parameters(parameters, gradients, learning_rate)

        ### END CODE HERE ###

        return loss, gradients, a[len(X) - 1]

    # GRADED FUNCTION: model

    def model(data, ix_to_char, char_to_ix, num_iterations=35000, n_a=50, dino_names=7, vocab_size=27, verbose=False):
        """
        Trains the model and generates dinosaur names.

        Arguments:
        data -- text corpus
        ix_to_char -- dictionary that maps the index to a character
        char_to_ix -- dictionary that maps a character to an index
        num_iterations -- number of iterations to train the model for
        n_a -- number of units of the RNN cell
        dino_names -- number of dinosaur names you want to sample at each iteration.
        vocab_size -- number of unique characters found in the text (size of the vocabulary)

        Returns:
        parameters -- learned parameters
        """

        # Retrieve n_x and n_y from vocab_size
        n_x, n_y = vocab_size, vocab_size

        # Initialize parameters
        parameters = initialize_parameters(n_a, n_x, n_y)

        # Initialize loss (this is required because we want to smooth our loss)
        loss = get_initial_loss(vocab_size, dino_names)

        # Build list of all dinosaur names (training examples).
        with open("data/dinos.txt") as f:
            examples = f.readlines()
        examples = [x.lower().strip() for x in examples]

        # Shuffle list of all dinosaur names
        np.random.seed(0)
        np.random.shuffle(examples)

        # Initialize the hidden state of your LSTM
        a_prev = np.zeros((n_a, 1))

        # Optimization loop
        for j in range(num_iterations):

            ### START CODE HERE ###

            # Set the index `idx` (see instructions above)
            idx = j % len(examples)

            # Set the input X (see instructions above)
            single_example = examples[idx]
            single_example_chars = [c for c in single_example]
            single_example_ix = [char_to_ix[c] for c in single_example_chars]
            X = [None] + single_example_ix

            # Set the labels Y (see instructions above)
            ix_newline = char_to_ix['\n']
            Y = X[1:] + [ix_newline]

            # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
            # Choose a learning rate of 0.01
            curr_loss, gradients, a_prev = ModelService.optimize(X, Y, a_prev, parameters)

            ### END CODE HERE ###

            # debug statements to aid in correctly forming X, Y
            if verbose and j in [0, len(examples) - 1, len(examples)]:
                print("j = ", j, "idx = ", idx, )
            if verbose and j in [0]:
                print("single_example =", single_example)
                print("single_example_chars", single_example_chars)
                print("single_example_ix", single_example_ix)
                print(" X = ", X, "\n", "Y =       ", Y, "\n")

            # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
            loss = smooth(loss, curr_loss)

            # Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
            if j % 2000 == 0:

                print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

                # The number of dinosaur names to print
                seed = 0
                for name in range(dino_names):
                    # Sample indices and print them
                    sampled_indices = ModelService.sample(parameters, char_to_ix, seed)
                    print_sample(sampled_indices, ix_to_char)

                    seed += 1  # To get the same result (for grading purposes), increment the seed by one.

                print('\n')

        return parameters