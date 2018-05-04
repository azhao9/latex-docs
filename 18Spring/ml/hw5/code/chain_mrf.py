import numpy as np

class ChainMRFPotentials:
    def __init__(self, data_file):
        with open(data_file) as reader:
            for line in reader:
                if len(line.strip()) == 0:
                    continue

                split_line = line.split(" ")
                try:
                    self._n = int(split_line[0])
                except ValueError:
                    raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                try:
                    self._k = int(split_line[1])
                except ValueError:
                    raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                break

            # create an "(n+1) by (k+1)" list for unary potentials
            self._potentials1 = [[-1.0] * ( self._k + 1) for n in range(self._n + 1)]
            # create a "2n by (k+1) by (k+1)" list for binary potentials
            self._potentials2 = [[[-1.0] * (self._k + 1) for k in range(self._k + 1)] for n in range(2 * self._n)]

            for line in reader:
                if len(line.strip()) == 0:
                    continue

                split_line = line.split(" ")

                if len(split_line) == 3:
                    try:
                        i = int(split_line[0])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                    try:
                        a = int(split_line[1])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                    if i < 1 or i > self._n:
                        raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
                    if a < 1 or a > self._k:
                        raise Exception("given k=" + str(self._k) + ", illegal value for a: " + str(a))
                    if self._potentials1[i][a] >= 0.0:
                        raise Exception("ill-formed energy file: duplicate keys: " + line)
                    self._potentials1[i][a] = float(split_line[2])
                elif len(split_line) == 4:
                    try:
                        i = int(split_line[0])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                    try:
                        a = int(split_line[1])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                    try:
                        b = int(split_line[2])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[2] + " to integer.")
                    if i < self._n + 1 or i > 2 * self._n - 1:
                        raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
                    if a < 1 or a > self._k or b < 1 or b > self._k:
                        raise Exception("given k=" + self._k + ", illegal value for a=" + str(a) + " or b=" + str(b))
                    if self._potentials2[i][a][b] >= 0.0:
                        raise Exception("ill-formed energy file: duplicate keys: " + line)
                    self._potentials2[i][a][b] = float(split_line[3])
                else:
                    continue

            # check that all of the needed potentials were provided
            for i in range(1, self._n + 1):
                for a in range(1, self._k + 1):
                    if self._potentials1[i][a] < 0.0:
                        raise Exception("no potential provided for i=" + str(i) + ", a=" + str(a))
            for i in range(self._n + 1, 2 * self._n):
                for a in range(1, self._k + 1):
                    for b in range(1, self._k + 1):
                        if self._potentials2[i][a][b] < 0.0:
                            raise Exception("no potential provided for i=" + str(i) + ", a=" + str(a) + ", b=" + str(b))

    def chain_length(self):
        return self._n

    def num_x_values(self):
        return self._k

    def potential(self, i, a, b = None):
        if b is None:
            if i < 1 or i > self._n:
                raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
            if a < 1 or a > self._k:
                raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a))
            return self._potentials1[i][a]

        if i < self._n + 1 or i > 2 * self._n - 1:
            raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
        if a < 1 or a > self._k or b < 1 or b > self._k:
            raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a) + " or b=" + str(b))
        return self._potentials2[i][a][b]


class SumProduct:

    def __init__(self, p):
        self._potentials = p
        self.n = p.chain_length()
        self.k = p.num_x_values()
        self.right_messages = -1*np.ones((self.k + 1, self.n + 1))
        self.left_messages = -1*np.ones((self.k + 1, self.n + 1))
        # add whatever data structures needed

    def right_message(self, x_i, value):
        if (x_i == self.n):
            raise Exception("can't send message to right of x_n")

        if (self.right_messages[value, x_i] != -1):
            return self.right_messages[value, x_i]

        if (x_i == 1):
            message = self._potentials.potential(1, value)
        else:
            f_n = x_i + self.n - 1
            f_n_potentials = [self._potentials.potential(f_n, k, value) for k in range(1, self.k+1)]
            previous_messages = [self.right_message(x_i-1, val) for val in range(1, self.k+1)]
            message = self._potentials.potential(x_i, value) * np.dot(f_n_potentials, previous_messages)

        self.right_messages[value, x_i] = message
        return message

    def left_message(self, x_i, value):
        if (x_i == 1):
            raise Exception("can't send message to left of x_1")

        if (self.left_messages[value, x_i] != -1):
            return self.left_messages[value, x_i]

        if (x_i == self.n):
            message = self._potentials.potential(self.n, value)
        else:
            f_n = x_i + self.n
            f_n_potentials = [self._potentials.potential(f_n, value, k) for k in range(1, self.k+1)]
            previous_messages = [self.left_message(x_i+1, val) for val in range(1, self.k+1)]
            message = self._potentials.potential(x_i, value) * np.dot(f_n_potentials, previous_messages)

        self.left_messages[value, x_i] = message
        return message

    def marginal_probability(self, x_i):
        # TODO: EDIT HERE
        # should return a python list of type float, with its length=k+1, and the first value 0

        # This code is used for testing only and should be removed in your implementation.
        # It creates a uniform distribution, leaving the first position 0
        '''
        result = [1.0 / (self._potentials.num_x_values())] * (self._potentials.num_x_values() + 1)
        result[0] = 0
        return result
        '''

        values = range(1, self.k+1)

        fi_xi = [self._potentials.potential(x_i, value) for value in values]

        if (x_i == 1):
            # f_1 -> x_1 and f_(n+1) -> x_1     
            all_left_messages = [self.left_message(2, value) for value in values]
            left_f = [[self._potentials.potential(1 + self.n, a, b) for a in values] for b in values]
            result = np.multiply(fi_xi, np.dot(all_left_messages, left_f))

        elif (x_i == self.n):
            # f_n -> x_n and f_(2n-1) -> x_n
            all_right_messages = [self.right_message(self.n-1, value) for value in values]
            right_f = [[self._potentials.potential(2*self.n-1, a, b) for a in values] for b in values]
            result = np.multiply(fi_xi, np.dot(right_f, all_right_messages))

        else:
            # f_i -> x_i, f_(n+i-1) -> x_i, f_(n+i) -> x_i
            all_left_messages = [self.left_message(x_i+1, value) for value in values]
            all_right_messages = [self.right_message(x_i-1, value) for value in values]
            left_f = [[self._potentials.potential(x_i + self.n, a, b) for a in values] for b in values]
            right_f = [[self._potentials.potential(x_i + self.n - 1, a, b) for a in values] for b in values]

            left_prod = np.dot(all_left_messages, left_f)
            right_prod = np.dot(right_f, all_right_messages)
            result = fi_xi * left_prod * right_prod

        result = np.insert(result, 0, 0)

        result /= np.sum(result)

        return result


class MaxSum:
    def __init__(self, p):
        self._potentials = p
        self._assignments = [0] * (p.chain_length() + 1)
        self.n = p.chain_length()
        self.k = p.num_x_values()
        self.right_messages = -1*np.ones((self.k + 1, self.n + 1))
        self.left_messages = -1*np.ones((self.k + 1, self.n + 1))

        # TODO: EDIT HERE
        # add whatever data structures needed

    def get_assignments(self):
        return self._assignments

    def right_message(self, x_i, value):
        if (x_i == self.n):
            raise Exception("can't send message to right of x_n")

        if (self.right_messages[value, x_i] != -1):
            return self.right_messages[value, x_i]

        if (x_i == 1):
            message = np.log(self._potentials.potential(1, value))
        else:
            f_n = x_i + self.n - 1
            log_f_n_potentials = [np.log(self._potentials.potential(f_n, k, value)) for k in range(1, self.k+1)]
            previous_messages = [self.right_message(x_i-1, val) for val in range(1, self.k+1)]
            message = np.log(self._potentials.potential(x_i, value)) + np.max(log_f_n_potentials + previous_messages)

        self.right_messages[value, x_i] = message
        return message

    def left_message(self, x_i, value):
        if (x_i == 1):
            raise Exception("can't send message to left of x_1")

        if (self.left_messages[value, x_i] != -1):
            return self.left_messages[value, x_i]

        if (x_i == self.n):
            message = np.log(self._potentials.potential(self.n, value))
        else:
            f_n = x_i + self.n
            log_f_n_potentials = [np.log(self._potentials.potential(f_n, value, k)) for k in range(1, self.k+1)]
            previous_messages = [self.left_message(x_i+1, val) for val in range(1, self.k+1)]
            message = np.log(self._potentials.potential(x_i, value)) + np.max(log_f_n_potentials + previous_messages)

        self.left_messages[value, x_i] = message
        return message

    def max_probability(self, x_i):
        values = range(1, self.k+1)

        max_j = -1
        max_prob = float('-inf')

        for j in range(1, self.k+1):
            log_fi_xi = np.log(self._potentials.potential(x_i, j))
            if (x_i == 1):
                # f_1 -> x_1 and f_(n+1) -> x_1     
                all_left_messages = [self.left_message(2, value) for value in values]
                log_left_f = [np.log(self._potentials.potential(1 + self.n, a, j)) for a in values] 
                result = log_fi_xi + np.amax(all_left_messages + log_left_f)

            elif (x_i == self.n):
                # f_n -> x_n and f_(2n-1) -> x_n
                all_right_messages = [self.right_message(self.n-1, value) for value in values]
                log_right_f = [np.log(self._potentials.potential(2*self.n-1, j, b)) for b in values] 
                result = log_fi_xi + np.amax(log_right_f + all_right_messages)

            else:
                # f_i -> x_i, f_(n+i-1) -> x_i, f_(n+i) -> x_i
                all_left_messages = [self.left_message(x_i+1, value) for value in values]
                all_right_messages = [self.right_message(x_i-1, value) for value in values]
                log_left_f = [np.log(self._potentials.potential(x_i + self.n, j, b)) for b in values]
                log_right_f = [np.log(self._potentials.potential(x_i + self.n - 1, a, j)) for a in values] 

                left_max = np.amax(all_left_messages + log_left_f)
                right_max = np.amax(log_right_f + all_right_messages)
                result = log_fi_xi + left_max + right_max

            if (result > max_prob):
                max_prob = result
                max_j = j

        self._assignments[x_i] = max_j

        return max_prob


    
