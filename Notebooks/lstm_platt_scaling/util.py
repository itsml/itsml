import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import (confusion_matrix, f1_score,
                             accuracy_score, recall_score, precision_score, roc_auc_score)

import pickle

class Data:
    
    def __init__(self, data_path, sequence_path=None):
        self.data_path = data_path
        self.sequence_path = sequence_path

    
    def std_data(self, train_df, test_df):
        """Standardise data.

            Standardises a data set tuple:

            Parameters
            ----------
            train_df : Pandas.Dataframe
                Training data set.
            test_df : Pandas.Dataframe
                Test data set.

            Returns
            -------
            tuple of Pandas.Dataframes
                Training and test data set.


        """
        # training data
        train_df['time_norm'] = train_df['time']
        cols_normalize = train_df.columns.difference(['id','time','RUL', 'label_1'])
        standard_scaler = preprocessing.StandardScaler()
        
        
        norm_train_df = pd.DataFrame(standard_scaler.fit_transform(train_df[cols_normalize]), 
                                     columns=cols_normalize, 
                                     index=train_df.index)
        join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
        train_df = join_df.reindex(columns = train_df.columns)

        # test data
        test_df['time_norm'] = test_df['time']
        norm_test_df = pd.DataFrame(standard_scaler.transform(test_df[cols_normalize]), 
                                    columns=cols_normalize, 
                                    index=test_df.index)
        test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
        test_df = test_join_df.reindex(columns = test_df.columns)
        test_df = test_df.reset_index(drop=True)

        return train_df, test_df
    
        
    def gen_sequence(self, id_df, seq_length, seq_cols):
        """ Generates sequences with a sliding window.
        
            Only sequences that meet the window-length are considered, no padding is used. This means for testing
            we need to drop those which are below the window-length.

            Parameters
            ----------
            id_df : Pandas.Dataframe
                Dataframe the sequences are created from.
            seq_length : int
                Length of the sequences.
            seq_cols : List of str
                Keys of the columns of the dataframe.

            Yields
            -------
            numpy.ndarray 
                Sequence based on the input dataframe.
        """
        # for one id I put all the rows in a single matrix
        data_matrix = id_df[seq_cols].values
        num_elements = data_matrix.shape[0]
        # Iterate over two lists in parallel.
        # For example id1 have 192 rows and sequence_length is equal to 50
        # so zip iterate over two following list of numbers (0,142),(50,192)
        # 0 50 -> from row 0 to row 50
        # 1 51 -> from row 1 to row 51
        # 2 52 -> from row 2 to row 52
        # ...
        # 141 191 -> from row 141 to 191
        
#         print("seq_len: {}\tnum_elements: {}".format(seq_length, num_elements))
        if num_elements == seq_length:
#             print(num_elements)
#             print(data_matrix[seq_length-1:seq_length, :].shape)
#             print(data_matrix[seq_length-1:seq_length, :])
            yield data_matrix[0:seq_length, :]
        else:
            for start, stop in zip(range(0, (num_elements-seq_length)), range(seq_length, num_elements)):
                yield data_matrix[start:stop, :]

    def gen_sequence_time(self, id_df, seq_length, seq_cols):
        """ Generates sequences of the time steps with a sliding window.
        
            Only sequences that meet the window-length are considered, no padding is used. This means for testing
            we need to drop those which are below the window-length.

            Parameters
            ----------
            id_df : Pandas.Dataframe
                Dataframe the sequences are created from.
            seq_length : int
                Length of the sequences.
            seq_cols : List of str
                Keys of the columns of the dataframe.

            Yields
            -------
            numpy.ndarray 
                Sequence of time steps based on the input dataframe.
        """
        # for one id I put all the rows in a single matrix
        data_matrix = id_df[seq_cols].values
        num_elements = data_matrix.shape[0]
        # Iterate over two lists in parallel.
        # For example id1 have 192 rows and sequence_length is equal to 50
        # so zip iterate over two following list of numbers (0,142),(49,191)
        # 0 49 -> from row 1 to row 50
        # 1 50 -> from row 2 to row 51
        # 2 51 -> from row 3 to row 53
        # ...
        # 141 191 -> from row 142 to 192
        if num_elements == seq_length:
            yield seq_length
        else:
            for start, stop in zip(range(0,(num_elements-seq_length)), range(seq_length, num_elements)):
                yield stop

    # function to generate labels
    def gen_labels(self, id_df, seq_length, label):
        """ Generates sequences of the labels with a sliding window.
        
            Only sequences that meet the window-length are considered, no padding is used. This means for testing
            we need to drop those which are below the window-length.

            Parameters
            ----------
            id_df : Pandas.Dataframe
                Dataframe the sequences are created from.
            seq_length : int
                Length of the sequences.
            label : str
                Key of the label.

            Yields
            -------
            numpy.ndarray 
                Sequence of labels based on the input dataframe.
        """
        data_array = id_df[label].values
        num_elements = data_array.shape[0]

        # num_elements-1? Because of range stop goes to num_elements-1, too
        if num_elements == seq_length:
            return data_array[seq_length-1:num_elements, :]
        else:
            return data_array[seq_length:num_elements, :]

    def create_sequences(self, train_df, test_df, target_label, sequence_cols,
                         val_ratio=0, sequence_length=50, save=True, random_seed=42):
        """Create sequences for use with RNNs.

            Creates a sequence for data and label for training set and test set:

            Parameters
            ----------
            train_df : Pandas.Dataframe
                Training data set.
            test_df : Pandas.Dataframe
                Test data set.
            target_label : str
                Label filtered for to create label sequence.
            sequence_cols : List of str
                Selected solumns for sequence generation.
            val_ratio : float
                Ratio of train set that is cut off as validation set
            sequence_length : int
                Length of the sequences.
            save : bool
                Save results to file.
            time : bool
                Add an extra df to output tuple with time of last datapoint in sequence.

            Returns
            -------
            dict of Pandas.Dataframes
                tuple of sequences for training and test data set.
                train_seq
                train_time
                train_label
                test_seq
                test_time
                test_label
                test_last_seq
                test_last_time
                test_last_label
        """
        if save:
            suffix_train = 'train'
            suffix_val = 'val'
            suffix_test = 'test'
            suffix_test_last = 'test_last'
        else:
            suffix_train = None
            suffix_val = None
            suffix_test = None
            suffix_test_last = None


        if val_ratio > 0:
        
            max_id = train_df["id"].max()
            train_length = int(max_id * (1 - val_ratio))
            
            rng = np.random.default_rng(random_seed)
            train_ids = rng.choice(np.arange(1, max_id+1), train_length, replace=False)

            train_df_cut =train_df[train_df['id'].isin(train_ids)]
            val_df = train_df[~train_df['id'].isin(train_ids)] # ~ is the 'complement' operator for pandas
            
            (seq_array, 
            time_array, 
            label_array) = self.create_filtered_sequences(train_df_cut, target_label, sequence_cols,
                                                         sequence_length=sequence_length, save_suffix=suffix_train)
            (seq_array_val, 
            time_array_val, 
            label_array_val) = self.create_filtered_sequences(val_df, target_label, sequence_cols,
                                                             sequence_length=sequence_length, save_suffix=suffix_val)
        else:
            (seq_array, 
            time_array, 
            label_array) = self.create_filtered_sequences(train_df, target_label, sequence_cols,
                                                         sequence_length=sequence_length, save_suffix=suffix_train)
        
        (seq_array_test,
        time_array_test,
        label_array_test) = self.create_filtered_sequences(test_df, target_label, sequence_cols,
                                                          sequence_length=sequence_length, save_suffix=suffix_test)
        (seq_array_test_last,
        time_array_test_last,
        label_array_test_last) = self.create_last_sequences(test_df, target_label, sequence_cols,
                                                           sequence_length=sequence_length, save_suffix=suffix_test_last)
        seq_dict = {'train_seq': seq_array,
                    'train_time': time_array,
                    'train_label': label_array,
                    'test_seq': seq_array_test,
                    'test_time': time_array_test,
                    'test_label': label_array_test,
                    'test_last_seq': seq_array_test_last,
                    'test_last_time': time_array_test_last,
                    'test_last_label': label_array_test_last}
        if val_ratio > 0:
            seq_dict.update({'val_seq': seq_array_val,
                             'val_time': time_array_val,
                             'val_label': label_array_val})
        return seq_dict
    
    def create_unfiltered_sequences(self, df, target_label, sequence_cols, sequence_length=50, save_suffix=None):
        """Create training sequences for use with LSTMs.

            Creates a sequence for data, time and label for data set disregarding length of possible sequences. 
            More error prone:

            Parameters
            ----------
            df : Pandas.Dataframe
                Data set.
            target_label : str
                Label filtered for to create label sequence.
            sequence_cols : List of str
                Selected solumns for sequence generation.
            sequence_length : int
                Length of the sequences.
            save_suffix : None or str
                Save results to file with save_suffix.
            time : bool
                Add an extra df to output tuple with time of last datapoint in sequence.

            Returns
            -------
            tupe of Pandas.Dataframes
                tuple of sequences of the data set.
                (seq_array, time_array, label_array)
        """
        
        # generate data sequences
        seq_gen = (list(self.gen_sequence(df[df['id']==id], sequence_length, sequence_cols)) 
                   for id in df['id'].unique())
        seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
        
        # generate time sequences
        time_gen = (list(self.gen_sequence_time(df[df['id']==id], sequence_length, sequence_cols)) 
                    for id in df['id'].unique())
        time_array = np.concatenate(list(time_gen)).astype(np.float32)
        
        # generate label sequences
        label_gen = [self.gen_labels(df[df['id']==id], sequence_length, [target_label]) 
                     for id in df['id'].unique()]
        label_array = np.concatenate(label_gen).astype(np.float32)
 
        if save_suffix is not None:
            np.save(self.sequence_path + 'seq_array_' + save_suffix + '.npy', seq_array)
            np.save(self.sequence_path + 'time_array_' + save_suffix + '.npy', time_array)
            np.save(self.sequence_path + 'label_array_' + save_suffix + '.npy', label_array)
        
        return seq_array, time_array, label_array
    
    def create_filtered_sequences(self, df, target_label, sequence_cols, sequence_length=50, save_suffix=None):
        """Creates sequences for use with LSTMs.

            Creates a sequence for data, time and label for a data set.
            Filters out ids with less entries than the sequence length:

            Parameters
            ----------
            df : Pandas.Dataframe
                Data set.
            target_label : str
                Label filtered for to create label sequence.
            sequence_cols : List of str
                Selected solumns for sequence generation.
            sequence_length : int
                Length of the sequences.
            save_suffix : None or str
                Save results to file with save_suffix.
            time : bool
                Add an extra df to output tuple with time of last datapoint in sequence.

            Returns
            -------
            tupe of Pandas.Dataframes
                tuple of sequences of the data set.
                (seq_array, time_array, label_array)
        """        
        # Filter ids        
        y_mask = [len(df[df['id']==id]) >= sequence_length for id in df['id'].unique()]
        ids = []
        all_ids = df['id'].unique()
        for i in range(len(all_ids)):
            if y_mask[i]:
                ids.append(all_ids[i])
        df_cropped = df[df['id'].isin(ids)]

        seq_gen =  (list(self.gen_sequence(df_cropped[df_cropped['id']==id], sequence_length,sequence_cols))
                    for id in df_cropped['id'].unique())
        seq_gen_list = list(seq_gen)
        seq_array = np.concatenate(seq_gen_list).astype(np.float32)

        # generate time sequences
        time_gen = (list(self.gen_sequence_time(df_cropped[df_cropped['id']==id], sequence_length, sequence_cols))
                    for id in df_cropped['id'].unique())
        time_array = np.concatenate(list(time_gen)).astype(np.float32)
        
        # generate label sequences
        label_gen = [self.gen_labels(df_cropped[df_cropped['id']==id], sequence_length, [target_label]) 
                     for id in df_cropped['id'].unique()]

        label_array = np.concatenate(label_gen).astype(np.float32)
        
        if save_suffix is not None:
            np.save(self.sequence_path + 'seq_array_' + save_suffix + '.npy', seq_array)
            np.save(self.sequence_path + 'time_array_' + save_suffix + '.npy', time_array)
            np.save(self.sequence_path + 'label_array_' + save_suffix + '.npy', label_array)
        
        return seq_array, time_array, label_array
    
    def create_last_sequences(self, df, target_label, sequence_cols, sequence_length=50, save_suffix=None):
        """Create test sequences with only last entries of series for use with LSTMs.

            Creates a sequence for data, time and label for last entries of each id in data set:

            Parameters
            ----------
            df : Pandas.Dataframe
                Data set.
            target_label : str
                Label filtered for to create label sequence.
            sequence_cols : List of str
                Selected solumns for sequence generation.
            sequence_length : int
                Length of the sequences.
            save_suffix : None or str
                Save results to file with save_suffix.

            Returns
            -------
            tupe of Pandas.Dataframes
                tuple of sequences of the data set.
                (seq_array, time_array, label_array)
        """
        # We pick the last sequence for each id in the data data
        seq_array = [df[df['id']==id][sequence_cols].values[-sequence_length:] 
                               for id in df['id'].unique() if len(df[df['id']==id]) >= sequence_length]
        seq_array = np.asarray(seq_array).astype(np.float32)
        
        # generate time sequences
        time_array = np.array([df[df['id']==id][sequence_cols].values[-sequence_length:].shape[0]
                                for id in df['id'].unique()]).astype(np.float32)
        
        # generate label sequences by masking with the ids of proper length
        y_mask = [len(df[df['id']==id]) >= sequence_length for id in df['id'].unique()]
        label_array = df.groupby('id')['label_1'].nth(-1)[y_mask].values
        label_array = label_array.reshape(label_array.shape[0],1).astype(np.float32)
            
        if save_suffix is not None:
            np.save(self.sequence_path + 'seq_array_' + save_suffix + '.npy', seq_array)
            np.save(self.sequence_path + 'time_array_' + save_suffix + '.npy', time_array)
            np.save(self.sequence_path + 'label_array_' + save_suffix + '.npy', label_array)

        return seq_array, time_array, label_array


class CMAPSS_Data(Data):
    
    def __init__(self, data_path, sequence_path=None):
        super().__init__(data_path, sequence_path)

    def load_data(self, data_idx=1, w=30, preprocess="std"):
        """Load C-MAPSS data from file.

            The corresponding C-MAPSS data is laoded from file according to index number:

            Parameters
            ----------
            data_idx : int
                Index of the data set.
            w : int
                Number of cycles before EOL. Is a specific engine going to fail within w cycles?
            preprocess : str
                Defines the preprocessing of the data set.

            Returns
            -------
            tuple of Pandas.Dataframes
                Training and test data set.


        """

        train_df = pd.read_csv(self.data_path + '/train_FD00' + str(data_idx) + '.txt', sep=' ', header=None)
        train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
        train_df.columns = ['id', 'time', 'setting_1', 'setting_2', 'setting_3', 's1', 's2', 's3', 's4', 's5',
                          's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17','s18',
                          's19', 's20', 's21']


        rul = pd.DataFrame(train_df.groupby('id')['time'].max()).reset_index()
        rul.columns = ['id', 'max']
        train_df = train_df.merge(rul, on=['id'], how='left')
        train_df['RUL'] = train_df['max'] - train_df['time']
        train_df.drop('max', axis=1, inplace=True)

        # generate label column for training data
        # for binary classification we will try to answer the question: 
        # is a specific engine going to fail within w cycles?

        train_df['label_1'] = np.where(train_df['RUL'] <= w, 1, 0 )


        test_df = pd.read_csv(self.data_path + '/test_FD00' + str(data_idx) + '.txt', sep=' ', header=None)
        test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
        test_df.columns = ['id', 'time', 'setting_1', 'setting_2', 'setting_3', 's1', 's2', 's3', 's4', 's5',
                          's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17','s18',
                          's19', 's20', 's21']


        test_y = pd.read_csv(self.data_path + '/RUL_FD00' + str(data_idx) + '.txt', sep=' ', header=None)
        test_y.drop(test_y.columns[[1]], axis=1, inplace=True)

        # Create target for the test data
        rul = pd.DataFrame(test_df.groupby('id')['time'].max()).reset_index()
        rul.columns = ['id', 'max']
        test_y.columns = ['target_RUL']
        test_y['id'] = test_y.index + 1
        test_y['max'] = rul['max'] + test_y['target_RUL']
        test_y.drop('target_RUL', axis=1, inplace=True)

        test_df = test_df.merge(test_y, on=['id'], how='left')
        test_df['RUL'] = test_df['max'] - test_df['time']
        test_df.drop('max', axis=1, inplace=True)

        # generate label column w for test data
        test_df['label_1'] = np.where(test_df['RUL'] <= w, 1, 0 )

        if preprocess is "std":
            std_train_df, std_test_df = self.std_data(train_df, test_df)
            
            return std_train_df, std_test_df
        
        else:
            return train_df, test_df
        
def evaluate_prediction(y_pred, label_array, title="Eval", do_auroc=True, verbose=True):
    """Evaluates a prediction.

        Parameters
        ----------
        y_pred : numpy.ndarray
            Array of predicted labels.
        label_array : numpy.ndarray
            Array of true labels.
        title: str
            Title string for output.
        do_auroc: Bool
            Do AUROC score in evaluation.
        verbose: Bool
            Print evaulation scores.

        Returns
        -------
        tuple of floats
            Evaluation scores.


    """
    
    cm = confusion_matrix(label_array, y_pred)

    tn, fp, fn, tp = list(map(lambda x: round(x, 3), cm.ravel()))
    

    # compute precision and recall
    accuracy = round(accuracy_score(label_array, y_pred), 3)
    precision = round(precision_score(label_array, y_pred, zero_division=0), 3)
    recall = round(recall_score(label_array, y_pred, zero_division=0), 3)
    f1 = round(f1_score(label_array, y_pred, zero_division=0), 3)
    
    if do_auroc:
        auroc = round(roc_auc_score(label_array, y_pred), 3)
    else:
        auroc= -1
    
    if verbose:
        print('\n === ' + title + ' === \n')
        print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
        print(cm)
        print ('tp: {}, fp: {}, tn: {}, fn: {}'.format(tp, fp, tn, fn))
        print( ' Accuracy: ', accuracy, '\n', 'Precision: ', precision, '\n', 'Recall: ', recall,
              '\n', 'F1-score:', f1, '\n', 'AUROC-score:', auroc  )
    
    return (tn, fp, fn, tp, accuracy, precision, recall, f1, auroc)

def evaluate_lstm(lstm, data_array, label_array, title="Eval", do_auroc=True, verbose=True):
    """Evaluates a classifiers prediction.

        Parameters
        ----------
        lstm : tensorflow.keras.models.Model
            Lstm model to evaluate.
        label_array : numpy.ndarray
            Array of true labels.
        title: str
            Title string for output.
        do_auroc: Bool
            Do AUROC score in evaluation.
        verbose: Bool
            Print evaulation scores.

        Returns
        -------
        tuple of floats
            Evaluation scores.


    """
    y_pred = lstm.predict_classes(data_array, batch_size=200)
    
    return evaluate_prediction(y_pred, label_array, title=title, do_auroc=do_auroc, verbose=verbose)

def newton_platt(deci, label, prior_1, prior_0, maxiter=100, minstep=1e-10):
    """Platt scaling algorithm.

        Fits a sigmoid to output of binary classifier:

        Parameters
        ----------
        deci : np.array 
            Array of classifier outputs.
        label : np.array 
            Array of booleans: is the example labeled +1?
        prior_1 : int
            Number of positive examples.
        prior_0 : int
            Number of negative examples.
        maxiter: int
            Maximum number of iterations
        minstep: float
            Minimum step taken in line search
            
        Returns
        -------
        A, B: tuple of floats
            Parameters of sigmoid.
    """
    
    # Parameter setting
    sigma = 1e-12   # Set to any value > 0
    
    # Construct initial values:
    #   target support in array t,
    #   initial function value in fval
    hiTarget = (prior_1 + 1.0) / (prior_1 +2.0)
    loTarget = 1 / (prior_0 + 2.0)
    length = prior_1 + prior_0 # total number of data
    
    t = np.zeros(length)
    
    for i in range(0, length):
        if label[i]:
            t[i] = hiTarget
        else:
            t[i] = loTarget
    
    A = 0.0
    B = np.log((prior_0 + 1.0) / (prior_1 + 1.0))
    fval = 0.0
    
    for i in range(0, length):
        fApB = deci[i] * A + B
        
        if fApB >= 0:
            fval += t[i] * fApB + np.log(1 + np.exp(-fApB))
        else:
            fval += (t[i] - 1) * fApB + np.log(1 + np.exp(fApB))
            
    for it in range(1, maxiter +1):
        # Update gradient and hessian (use H' = H + sigma I)
        h11 = sigma
        h22 = sigma
        h21 = 0.0
        g1 = 0.0
        g2 = 0.0
        
        for i in range(0, length):
            fApB = deci[i] * A + B
            if fApB >= 0:
                p = np.exp(-fApB) / (1.0 + np.exp(-fApB))
                q = 1.0 / (1.0 + np.exp(-fApB))
            else:
                p = 1.0 / (1.0 + np.exp(fApB))
                q = np.exp(fApB) / (1.0 + np.exp(fApB))
            d2 = p * q
            h11 += deci[i]**2 * d2
            h22 += d2
            h21 += deci[i] * d2
            
            d1 = t[i] - p
            g1 += deci[i] * d1
            g2 += d1
        
        # Stopping criteria
        if abs(g1) < 1e-5 and abs(g2) < 1e-5:
            break
        
        # Compute modified Newton directions
        det = h11 * h22 - h21**2
        dA = -(h22 * g1 - h21 * g2)/det
        dB = -(-h21 * g1 + h11 * g2) /det
        gd = g1 * dA + g2 *dB
        
        stepsize = 1
        
        while(stepsize >= minstep):
            # Line Search
            newA = A + stepsize * dA
            newB = B + stepsize * dB
            newf = 0.0
            
            for i in range(0,length):
                fApB = deci[i] * newA + newB
                if fApB >= 0:
                    newf += t[i] * fApB + np.log(1 + np.exp(-fApB))
                else:
                    newf += (t[i] - 1) * fApB + np.log(1 + np.exp(fApB))
            if newf < fval + 0.0001 * stepsize * gd:
                A = newA
                B = newB
                fval = newf
                break # sufficient decrease satisfied
            else:
                stepsize /= 2.0
            
        if stepsize < minstep:
            print('Line search fails (stepsize: {})'.format(stepsize))
            break
    
    if it >= maxiter:
        print('Reaching maximum iterations')

    return A, B # sigmoid parameters

def platt_sigmoid(a, b, x):
    """Platt sigmoid function to obtain probabilities.

        Parameters
        ----------
        a : float 
            Sigmoid parameter.
        b : float 
            Sigmoid parameter
        x : float
            data input.

        Returns
        -------
        p: float
            Probability estimate.
    """
    return 1/ (1 + np.exp(x * a + b))

def predict_platt_probs(a, b, xs):
    """Platt sigmoid function to obtain probabilities.

        Parameters
        ----------
        a : float 
            Sigmoid parameter.
        b : float 
            Sigmoid parameter
        xs : np.array
            data input.

        Returns
        -------
        p: np.array
            Probability estimates.
    """    
    for x in xs:
        platt_sigmoid(a, b, x)
    return [platt_sigmoid(a, b, x) for x in xs]
