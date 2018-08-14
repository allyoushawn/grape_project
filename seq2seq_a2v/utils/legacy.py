from .build_graph_utils import transpose_batch_time
from tensorflow.contrib.rnn import LayerNormBasicLSTMCell, LSTMCell, GRUCell


def convert_raw_rnn_ta_to_tensor(ta):
    return transpose_batch_time(ta.stack())


def build_policy_raw_rnn(hyper_parms, batch_size):

    rnn_inputs = hyper_parms['model_input']
    policy_rnn_cell_num = hyper_parms['policy_rnn_cell_num']
    policy_rnn_type = hyper_parms['policy_rnn_type']
    sequence_length = hyper_parms['sequence_len']
    assigned_seg_act = hyper_parms['assigned_seg_act']
    policy_rnn_layer_num = hyper_parms['policy_rnn_layer_num']
    reproduce_policy = hyper_parms['reproduce_policy']
    greedy_policy = hyper_parms['greedy_policy']

    cells = []
    for _ in range(policy_rnn_layer_num):
        if policy_rnn_type == 'gru':
            rnn_cell = GRUCell(policy_rnn_cell_num)
        elif policy_rnn_type == 'lstm':
            #rnn_cell = LayerNormBasicLSTMCell(policy_rnn_cell_num)
            rnn_cell = LSTMCell(policy_rnn_cell_num)
        else:
            raise ValueError('RNN type should be LSTM or GRU')
        cells.append(rnn_cell)
    cell = tf.contrib.rnn.MultiRNNCell(cells)
    cell = PolicyRNNCell(cell)

    inputs = transpose_batch_time(rnn_inputs)
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    inputs_ta = inputs_ta.unstack(inputs)

    seg_act = transpose_batch_time(assigned_seg_act)
    seg_act_ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    seg_act_ta = seg_act_ta.unstack(seg_act)


    loop_state_ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    def loop_fn(time, cell_output, cell_state, loop_state):
        #check whether is initial condition
        if cell_output is None:  # time == 0
            next_cell_state = cell.zero_state(batch_size, tf.float32)
        else:
            next_cell_state = cell_state
        #check whether finished
        elements_finished = (time >= tf.cast(sequence_length, tf.int32))
        finished = tf.reduce_all(elements_finished)

        #decide action
        if cell_output is None:
            next_loop_state = loop_state_ta
        else:
            action = tf.cond(reproduce_policy,
                lambda: seg_act_ta.read(time),
                lambda: tf.multinomial(
                 tf.log(cell_output), 1, output_dtype=tf.int32))

            action = tf.cond(greedy_policy,
                lambda: tf.expand_dims(
                         tf.argmax(cell_output, axis=1, output_type=tf.int32), 1),
                lambda: action)

        next_input = tf.cond(finished,
            lambda: tf.zeros([batch_size, rnn_inputs.get_shape()[-1]],
                             dtype=tf.float32),
            lambda: inputs_ta.read(time))

        emit_output = cell_output  # == None for time == 0

        #writing the action into loop state
        if cell_output == None: # time == 0
            next_loop_state = loop_state_ta
        else:
            next_loop_state = loop_state.write(time - 1, action)

        return (elements_finished, next_input, next_cell_state,
                emit_output, next_loop_state)

    outputs_ta, _, loop_state_ta = tf.nn.raw_rnn(cell, loop_fn)
    outputs = convert_raw_rnn_ta_to_tensor(outputs_ta)
    sampled_actions = convert_raw_rnn_ta_to_tensor(loop_state_ta)


    return outputs, sampled_actions
