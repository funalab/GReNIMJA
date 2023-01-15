import torch
import torch.nn as nn
from typing import Tuple, List
from lstm2d_cell import LSTM2dCell
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)


class LSTM2d(nn.Module):
    """
    2D-LSTM sequence-to-sequence (2D-seq2seq) model.

    Based on the paper
        "Towards two-dimensional sequence to sequence model in neural machine translation."
        Bahar, Parnia, Christopher Brix, and Hermann Ney
        arXiv preprint arXiv:1810.03975 (2018)

    Args:
        embed_dim: dimension of embedding vectors
        state_dim_2d: dimension of the hidden / cell state of the 2d-LSTM cells
        encoder_state_dim: dimension of the hidden / cell state of the bidirectional encoder LSTM

        input_vocab_size: size of the input vocabulary (i.e. number of embedding vectors in the source language)
        output_vocab_size: size of the output vocabulary (i.e. number of embedding vectors in the target language)

        device: the device (CPU / GPU) to run all computations on / store tensors on

        max_output_len: the maximum number of tokens to generate for an output sequence in inference mode until an
                        <eos> token is generated

        bos_token: the token (index) representing the beginning of a sentence in the output vocabulary
        eos_token: the token (index) representing the end of a sentence in the output vocabulary
    """
    name = "lstm2d-plain"

    def __init__(self, future_dim, state_dim_2d, device):
        super(LSTM2d, self).__init__()

        self.future_dim = future_dim
        self.state_dim_2d = state_dim_2d
        #self.max_batch_size = max_batch_size
        self.device = device

        #self.embedding_dropout = nn.Dropout(p=dropout_p)

        # input to the 2d-cell is a concatenation of the hidden encoder states h_j and the embedded output tokens y_i-1
        #cell_input_dim = 2 * encoder_state_dim + embed_dim  # 2*encoder_state_dim since it's bidirectional
        cell_input_dim = 2 * future_dim  # 2*encoder_state_dim since it's bidirectional
        self.cell2d = LSTM2dCell(cell_input_dim, self.state_dim_2d, device=self.device)

        # final output layer for next predicted token
        #self.logits_dropout = nn.Dropout(p=dropout_p)
        #self.logits = nn.Linear(in_features=self.state_dim_2d, out_features=2)
        #self.loss_function = torch.nn.CrossEntropyLoss(reduction='sum').to(self.device)

        # the encoder LSTM goes over the input sequence x and provides the hidden states h_j for the 2d-LSTM
        #self.encoder = nn.LSTM(input_size=embed_dim, hidden_size=encoder_state_dim, bidirectional=True).to(self.device)

    def forward(self, x, y):
        """
        Runs the complete forward propagation for the 2d-LSTM with known target tokens (i.e. using teacher forcing) in
        an O(input_seq_len + output_seq_len) algorithm.

        Args:
            x: (input_seq_len x batch) input tokens (indices in range [0, input_vocab_size))
            x_lengths: (batch) lengths of the input sequences, used for masking
            y: (output_seq_len x batch) correct output tokens (indices in range [0, output_vocab_size))

        Note:
            - it is assumed that the last token of the input (x) is an <EOS> token
            - it is assumed that the last token of the targets (y) is an <EOS> token

        Returns:
            y_pred: (output_seq_len x batch x output_vocab_size)
                predicted output sequence (logits for output_vocab_size)
        """

        x = torch.transpose(x, 0, 1)
        y = torch.transpose(y, 0, 1)

        batch_size = x.size()[1]
        input_seq_len = x.size()[0]
        output_seq_len = y.size()[0]


        # store hidden and cell states, at the beginning filled with zeros
        states_s = torch.zeros(input_seq_len + 1, output_seq_len + 1, batch_size, self.state_dim_2d, device=self.device)
        states_c = torch.zeros(input_seq_len + 1, output_seq_len + 1, batch_size, self.state_dim_2d, device=self.device)

        for diagonal_num in range(input_seq_len + output_seq_len - 1):
            # calculate the indices for input / states / etc. for this diagonal
            (ver_from, ver_to), (hor_from, hor_to) = LSTM2d.__calculate_input_ranges(diagonal_num=diagonal_num,
                                                                                     input_seq_len=input_seq_len,
                                                                                     output_seq_len=output_seq_len)
            ver_state_ranges, hor_state_ranges, diag_ranges = LSTM2d.__calculate_state_ranges((ver_from, ver_to),
                                                                                              (hor_from, hor_to))
            ver_range_x, ver_range_y = ver_state_ranges
            hor_range_x, hor_range_y = hor_state_ranges
            diag_range_x, diag_range_y = diag_ranges

            # flip the output range so we take the inputs in the right order corresponding to the input range
            # Note: the 2d-cell with smallest source-position (horizontally) and largest target-position (vertically) is
            # the first cell in the diagonal!
            input_range = list(range(ver_from, ver_to))
            output_range = list(reversed(range(hor_from, hor_to)))
            diagonal_len = len(input_range)  # always == len(output_range)

            # calculate x input for this diagonal
            # treat diagonal as though it was a larger batch and reshape inputs accordingly
            new_batch_size = diagonal_len * batch_size

            '''
            max_diagonal = int(self.max_batch_size / batch_size)
            p = int(diagonal_len / max_diagonal)
            q = diagonal_len % max_diagonal

            print('p:', str(p))
            print('q:', str(q))
            print('max_diagonal: ', str(max_diagonal))

            if p != 0:
                new_batch_size = self.max_batch_size

            for batch in range(p):
                print(str(batch*self.max_batch_size) + '--' + str((batch+1)*self.max_batch_size))
                h_current = x[input_range[batch*max_diagonal:(batch+1)*max_diagonal], :, :].view(self.max_batch_size, self.future_dim)
                y_current = y[output_range[batch*max_diagonal:(batch+1)*max_diagonal], :, :].view(self.max_batch_size, self.future_dim)
                x_current = torch.cat([h_current, y_current], dim=-1)  # shape (batch*diagonal_len x input_dim)

                # calculate previous hidden & cell states for this diagonal
                s_prev_hor = states_s[hor_range_x, hor_range_y, batch*self.max_batch_size:(batch+1)*self.max_batch_size, :].view(new_batch_size, self.state_dim_2d)
                c_prev_hor = states_c[hor_range_x, hor_range_y, batch*self.max_batch_size:(batch+1)*self.max_batch_size, :].view(new_batch_size, self.state_dim_2d)
                s_prev_ver = states_s[ver_range_x, ver_range_y, batch*self.max_batch_size:(batch+1)*self.max_batch_size, :].view(new_batch_size, self.state_dim_2d)
                c_prev_ver = states_c[ver_range_x, ver_range_y, batch*self.max_batch_size:(batch+1)*self.max_batch_size, :].view(new_batch_size, self.state_dim_2d)

                # run batched computation for this diagonal
                c_next, s_next = self.cell2d.forward(x_current, s_prev_hor, s_prev_ver, c_prev_hor, c_prev_ver)

                # separate batch and diagonal_len again so we can store them accordingly
                c_next = c_next.view(diagonal_len, batch_size, self.state_dim_2d)
                s_next = s_next.view(diagonal_len, batch_size, self.state_dim_2d)

                # store new hidden and cell states at the right indices for the next diagonal(s) to use
                states_s[diag_range_x, diag_range_y, :, :] = s_next
                states_c[diag_range_x, diag_range_y, :, :] = c_next

            if q != 0:
                h_current = x[input_range, -new_batch_size:, :].view(
                    new_batch_size, self.future_dim)
                y_current = y[output_range, -new_batch_size:, :].view(
                    new_batch_size, self.future_dim)
                x_current = torch.cat([h_current, y_current], dim=-1)  # shape (batch*diagonal_len x input_dim)

                # calculate previous hidden & cell states for this diagonal
                s_prev_hor = states_s[hor_range_x, hor_range_y, -new_batch_size:, :].view(new_batch_size,
                                                                                          self.state_dim_2d)
                c_prev_hor = states_c[hor_range_x, hor_range_y, -new_batch_size:, :].view(new_batch_size,
                                                                                          self.state_dim_2d)
                s_prev_ver = states_s[ver_range_x, ver_range_y, -new_batch_size:, :].view(new_batch_size,
                                                                                          self.state_dim_2d)
                c_prev_ver = states_c[ver_range_x, ver_range_y, -new_batch_size:, :].view(new_batch_size,
                                                                                          self.state_dim_2d)

                # run batched computation for this diagonal
                c_next, s_next = self.cell2d.forward(x_current, s_prev_hor, s_prev_ver, c_prev_hor, c_prev_ver)

                # separate batch and diagonal_len again so we can store them accordingly
                c_next = c_next.view(diagonal_len, new_batch_size, self.state_dim_2d)
                s_next = s_next.view(diagonal_len, batch_size, self.state_dim_2d)

                # store new hidden and cell states at the right indices for the next diagonal(s) to use
                states_s[diag_range_x, diag_range_y, :, :] = s_next
                states_c[diag_range_x, diag_range_y, :, :] = c_next
            
            '''
            h_current = x[input_range, :, :].view(new_batch_size, self.future_dim)
            y_current = y[output_range, :, :].view(new_batch_size, self.future_dim)
            x_current = torch.cat([h_current, y_current], dim=-1)  # shape (batch*diagonal_len x input_dim)

            # calculate previous hidden & cell states for this diagonal
            s_prev_hor = states_s[hor_range_x, hor_range_y, :, :].view(new_batch_size, self.state_dim_2d)
            c_prev_hor = states_c[hor_range_x, hor_range_y, :, :].view(new_batch_size, self.state_dim_2d)
            s_prev_ver = states_s[ver_range_x, ver_range_y, :, :].view(new_batch_size, self.state_dim_2d)
            c_prev_ver = states_c[ver_range_x, ver_range_y, :, :].view(new_batch_size, self.state_dim_2d)

            # run batched computation for this diagonal
            c_next, s_next = self.cell2d.forward(x_current, s_prev_hor, s_prev_ver, c_prev_hor, c_prev_ver)

            # separate batch and diagonal_len again so we can store them accordingly
            c_next = c_next.view(diagonal_len, batch_size, self.state_dim_2d)
            s_next = s_next.view(diagonal_len, batch_size, self.state_dim_2d)

            # store new hidden and cell states at the right indices for the next diagonal(s) to use
            states_s[diag_range_x, diag_range_y, :, :] = s_next
            states_c[diag_range_x, diag_range_y, :, :] = c_next
        '''
        for x_hor in range(1, input_seq_len + 1):
            for y_ver in range(1, output_seq_len + 1):
                h_current = x[x_hor - 1, :, :]
                y_current = y[y_ver - 1, :, :]
                x_current = torch.cat([h_current, y_current], dim=-1)  # shape (batch*diagonal_len x input_dim)

                s_prev_hor = states_s[[x_hor-1], [y_ver], :, :].view(-1, self.state_dim_2d)
                c_prev_hor = states_c[[x_hor-1], [y_ver], :, :].view(-1, self.state_dim_2d)
                s_prev_ver = states_s[[x_hor], [y_ver-1], :, :].view(-1, self.state_dim_2d)
                c_prev_ver = states_c[[x_hor], [y_ver-1], :, :].view(-1, self.state_dim_2d)

                # run batched computation for this diagonal
                c_next, s_next = self.cell2d.forward(x_current, s_prev_hor, s_prev_ver, c_prev_hor, c_prev_ver)

                # store new hidden and cell states at the right indices for the next diagonal(s) to use
                states_s[x_hor, y_ver, :, :] = s_next
                states_c[x_hor, y_ver, :, :] = c_next
        '''

        # for the prediction, take the last (valid, non-padded) column of states and all but the first (1:) row
        states_for_pred = states_s[input_seq_len, output_seq_len, :, :]
        # states_for_pred = self.logits_dropout.forward(states_for_pred)

        # y_pred = self.logits.forward(states_for_pred)   # shape (output_seq_len x batch x output_vocab_size)
        return states_for_pred

        # run the inputs through the bidirectional encoder LSTM and use the hidden states for further processing

        #return self.__training_forward(h=x, y=y)


    def __training_forward(self, h, y):
        """
        Optimized implementation of the 2D-LSTM forward pass at training time, where the correct tokens y are known in
        advance.
        Processes the input in a diagonal-wise fashion, as described in the paper
            Handwriting Recognition with Large Multidimensional Long Short-Term Memory Recurrent Neural Networks
            by Voigtlaender et. al.

        Args:
            h: (input_seq_len x batch x 2*encoder_state_dim) hidden states of bidirectional encoder LSTM
                important: in training mode, the length of all source sequences in a batch must be of the same length
                    (i.e. no padding for the horizontal dimension, all sequences have length exactly input_seq_len)
            h_lengths: (batch) lengths of the input sequences in the batch (the rest is padding)
            y: (output_seq_len x batch) correct output tokens (indices in range [0, output_vocab_size))

        Returns:
            y_pred: (output_seq_len x batch x output_vocab_size)
                predicted output sequence (logits for output_vocab_size)
        """
        batch_size = h.size()[1]
        input_seq_len = h.size()[0]
        output_seq_len = y.size()[0]

        # store hidden and cell states, at the beginning filled with zeros
        states_s = torch.zeros(input_seq_len+1, output_seq_len+1, batch_size, self.state_dim_2d, device=self.device)
        states_c = torch.zeros(input_seq_len+1, output_seq_len+1, batch_size, self.state_dim_2d, device=self.device)

        #my code (cとsを一つづつ計算していく方法(メモリが足りないため))

        for x_hor in range(1, input_seq_len+1):
            for y_ver in range(1, output_seq_len+1):

                h_current = h[x_hor-1, :, :]
                y_current = y[y_ver-1, :, :]
                x_current = torch.cat([h_current, y_current], dim=-1)  # shape (batch*diagonal_len x input_dim)

                s_prev_hor = states_s[[x_hor-1], [y_ver], :, :].view(-1, self.state_dim_2d)
                c_prev_hor = states_c[[x_hor-1], [y_ver], :, :].view(-1, self.state_dim_2d)
                s_prev_ver = states_s[[x_hor], [y_ver-1], :, :].view(-1, self.state_dim_2d)
                c_prev_ver = states_c[[x_hor], [y_ver-1], :, :].view(-1, self.state_dim_2d)

                # run batched computation for this diagonal
                c_next, s_next = self.cell2d.forward(x_current, s_prev_hor, s_prev_ver, c_prev_hor, c_prev_ver)

                # store new hidden and cell states at the right indices for the next diagonal(s) to use
                states_s[x_hor, y_ver, :, :] = s_next
                states_c[x_hor, y_ver, :, :] = c_next


        '''
        for diagonal_num in range(input_seq_len + output_seq_len - 1):
            # calculate the indices for input / states / etc. for this diagonal
            (ver_from, ver_to), (hor_from, hor_to) = LSTM2d.__calculate_input_ranges(diagonal_num=diagonal_num,
                                                                                     input_seq_len=input_seq_len,
                                                                                     output_seq_len=output_seq_len)
            ver_state_ranges, hor_state_ranges, diag_ranges = LSTM2d.__calculate_state_ranges((ver_from, ver_to),
                                                                                              (hor_from, hor_to))
            ver_range_x, ver_range_y = ver_state_ranges
            hor_range_x, hor_range_y = hor_state_ranges
            diag_range_x, diag_range_y = diag_ranges


            # flip the output range so we take the inputs in the right order corresponding to the input range
            # Note: the 2d-cell with smallest source-position (horizontally) and largest target-position (vertically) is
            # the first cell in the diagonal!
            input_range = list(range(ver_from, ver_to))
            output_range = list(reversed(range(hor_from, hor_to)))
            diagonal_len = len(input_range)  # always == len(output_range)

            # calculate x input for this diagonal
            # treat diagonal as though it was a larger batch and reshape inputs accordingly
            new_batch_size = diagonal_len * batch_size

            h_current = h[input_range, :, :].view(new_batch_size, self.future_dim)
            y_current = y[output_range, :, :].view(new_batch_size, self.future_dim)
            x_current = torch.cat([h_current, y_current], dim=-1)   # shape (batch*diagonal_len x input_dim)

            # calculate previous hidden & cell states for this diagonal
            s_prev_hor = states_s[hor_range_x, hor_range_y, :, :].view(new_batch_size, self.state_dim_2d)
            c_prev_hor = states_c[hor_range_x, hor_range_y, :, :].view(new_batch_size, self.state_dim_2d)
            s_prev_ver = states_s[ver_range_x, ver_range_y, :, :].view(new_batch_size, self.state_dim_2d)
            c_prev_ver = states_c[ver_range_x, ver_range_y, :, :].view(new_batch_size, self.state_dim_2d)

            # run batched computation for this diagonal
            c_next, s_next = self.cell2d.forward(x_current, s_prev_hor, s_prev_ver, c_prev_hor, c_prev_ver)

            # separate batch and diagonal_len again so we can store them accordingly
            c_next = c_next.view(diagonal_len, batch_size, self.state_dim_2d)
            s_next = s_next.view(diagonal_len, batch_size, self.state_dim_2d)

            # store new hidden and cell states at the right indices for the next diagonal(s) to use
            states_s[diag_range_x, diag_range_y, :, :] = s_next
            states_c[diag_range_x, diag_range_y, :, :] = c_next
            print(states_s)
            '''
        # for the prediction, take the last (valid, non-padded) column of states and all but the first (1:) row
        states_for_pred = states_s[input_seq_len, output_seq_len, :, :]
        #states_for_pred = self.logits_dropout.forward(states_for_pred)

        #y_pred = self.logits.forward(states_for_pred)   # shape (output_seq_len x batch x output_vocab_size)
        return states_for_pred

    def loss(self, y_pred, y_target):
        """
        Returns the cross entropy loss value for the given predictions and targets, ignoring <pad>-targets.
        Args:
            y_pred: (output_seq_len x batch x output_vocab_size) predicted output sequence (float logits)
            y_target: (output_seq_len x batch) target output tokens (long indices into output_vocab_size)

        Returns: () scalar-tensor representing the cross-entropy loss between y_pred and y_target
        """
        y_pred = y_pred.to(self.device)
        y_target = y_target.to(self.device)
        return self.loss_function(y_pred.view(-1, self.output_vocab_size), y_target.view(-1))

    def padded_loss(self, y_pred, y_target):
        """
        Returns the cross entropy loss value for the given predictions and targets, ignoring <pad>-targets,
        and expanding the prediction
        Args:
            y_pred: (predicted_seq_len x batch x output_vocab_size) predicted output sequence (float logits)
            y_target: (target_seq_len x batch) target output tokens (long indices into output_vocab_size)

        Returns: () scalar-tensor representing the cross-entropy loss between y_pred and y_target
        """
        y_pred = y_pred.to(self.device)
        y_target = y_target.to(self.device)

        predicted_seq_len = y_pred.size()[0]
        target_seq_len, batch = y_target.size()
        diff = predicted_seq_len - target_seq_len
        if diff > 0:      # pad the target
            padding = torch.ones(diff, batch, dtype=y_target.dtype, device=self.device) * self.pad_token
            y_target = torch.cat([y_target, padding], dim=0)
        elif diff < 0:    # pad the prediction
            padding = torch.zeros(abs(diff), batch, self.output_vocab_size, dtype=y_pred.dtype, device=self.device)
            y_pred = torch.cat([y_pred, padding], dim=0)
        return self.loss(y_pred, y_target)

    def predict(self, x, x_lengths):
        """
        Runs the complete forward propagation for the 2d-LSTM with unknown target tokens (inference), in
        an O(input_seq_len * output_seq_len) algorithm.

        Args:
            x: (input_seq_len x batch) input tokens (indices in range [0, input_vocab_size))
            x_lengths: (batch) lengths of the input sequences, used for masking

        Note:
            - it is assumed that the last token of the input (x) is an <EOS> token

        Returns:
            y_pred: (output_seq_len x batch x output_vocab_size) predictions (logits) for the output sequence,
             where output_seq_len <= max_output_len (depending on when the model predicts <eos> for each sequence),
             zero-padded for sequences in the batch that were <eos>-ed by the model before iteration # output_seq_len
        """
        x = x.to(self.device)
        x_lengths = x_lengths.to(self.device)

        # run the inputs through the bidirectional encoder LSTM and use the hidden states for further processing
        h = self.__encoder_lstm(x, x_lengths)
        h_lengths = x_lengths

        batch_size = h.size()[1]
        input_seq_len = h.size()[0]

        # initialize y to (embedded) start tokens
        y_i = torch.tensor([self.bos_token], dtype=torch.long, device=self.device).repeat(batch_size)
        y_i_emb = self.output_embedding.forward(y_i)

        # hidden states and cell states at previous vertical step i-1
        s_prev_i = torch.zeros(input_seq_len, batch_size, self.state_dim_2d, device=self.device)
        c_prev_i = torch.zeros(input_seq_len, batch_size, self.state_dim_2d, device=self.device)

        # result tensor (will later be truncated to the longest generated sequence in the batch in the first dimension)
        y_pred = torch.zeros(self.max_output_len, batch_size, self.output_vocab_size, device=self.device)

        # create horizontal mask tensor based on h_lengths to handle padding
        #hor_mask = torch.zeros(batch_size, input_seq_len, device=self.device)
        hor_mask = torch.ones(batch_size, input_seq_len, device=self.device)
        #for i in range(batch_size):
        #    hor_mask[i, :h_lengths[i]] = 1

        # go through each decoder output step, until either the maximum length is reached or all sentences are <eos>-ed
        i = 0
        active_indices = torch.tensor(list(range(batch_size)), device=self.device)
        while i < self.max_output_len and len(active_indices) > 0:
            num_seq_left = active_indices.size()[0]

            # initialize previous horizontal hidden state and cell state
            s_prev_hor = torch.zeros(num_seq_left, self.state_dim_2d, device=self.device)
            c_prev_hor = torch.zeros(num_seq_left, self.state_dim_2d, device=self.device)

            for j in range(input_seq_len):
                # input to 2d-cell is concatenation of encoder hidden state h_j and last generated token y_i
                h_j = h[j, active_indices, :]
                x_j = torch.cat([h_j, y_i_emb], dim=-1)     # shape (num_seq_len x input_dim)

                s_prev_ver = s_prev_i[j, active_indices, :]
                c_prev_ver = c_prev_i[j, active_indices, :]

                # both of shape (num_seq_left x state_dim_2d)
                c_next_hor, s_next_hor = self.cell2d.forward(x_j, s_prev_hor, s_prev_ver, c_prev_hor, c_prev_ver)

                # apply the mask (accounting for different input sequence lengths)
                mask = hor_mask[active_indices, j].view(-1, 1)  # broadcasts over cell_state_dim dimension
                s_prev_hor = (1 - mask) * s_prev_hor + mask * s_next_hor
                c_prev_hor = (1 - mask) * c_prev_hor + mask * c_next_hor

                s_prev_i[j, active_indices, :] = s_prev_hor
                c_prev_i[j, active_indices, :] = c_prev_hor

            # obtain next predicted token
            y_pred_i = self.logits.forward(s_prev_hor)  # (num_seq_left x output_vocab_size)
            y_pred[i, active_indices, :] = y_pred_i

            # remove sentences from the batch if the argmax prediction is an <eos> token
            # no value is equal to eos_token
            index_map = torch.ones(batch_size, dtype=torch.long, device=self.device) + self.eos_token
            argmax_tokens = torch.argmax(y_pred_i, dim=-1)          # (num_seq_left)
            index_map[active_indices] = argmax_tokens               # set the correct num_seq_left predictions

            # re-calculate the indices into the batch which are still active
            eosed_sequences = index_map.eq(self.eos_token)
            was_active_mask = torch.zeros(batch_size, dtype=torch.uint8, device=self.device)
            was_active_mask[active_indices] = 1

            active_indices = ((eosed_sequences == 0) * was_active_mask).nonzero().view(-1)
            assert active_indices.size()[0] == num_seq_left - eosed_sequences.sum().item()

            # next generated token embedding
            active_indices_into_current_seqs = (argmax_tokens.eq(self.eos_token) == 0).nonzero().view(-1)
            y_i_emb = self.output_embedding.forward(argmax_tokens[active_indices_into_current_seqs])
            i += 1

        # truncate to longest generated sequence (i <= self.max_output_len) (will be zero-padded)
        y_pred = y_pred[:i, :, :]
        return y_pred

    def __encoder_lstm(self, x, x_lengths):
        """
        Runs the bidirectional encoder LSTM on the input sequence to obtain the hidden states h_j.
        Args:
            x: (input_seq_len x batch) input tokens (indices in range [0, input_vocab_size))
            x_lengths: (batch) lengths of the (unpadded) input sequences, used for handling padding

        Returns:
            h: (input_seq_len x batch x 2*encoder_state_dim) hidden states of bidirectional encoder LSTM
        """
        embedded_x = self.input_embedding.forward(x)        # (input_seq_len x batch x embed_dim)
        embedded_x = self.embedding_dropout.forward(embedded_x)

        # pack and unpack the padded batch for the encoder
        packed_x = nn.utils.rnn.pack_padded_sequence(embedded_x, x_lengths)
        h, _ = self.encoder.forward(packed_x)               # (input_seq_len x batch x 2*encoder_state_dim)
        unpacked_h, _ = nn.utils.rnn.pad_packed_sequence(h)

        return unpacked_h

    @staticmethod
    def __calculate_input_ranges(diagonal_num: int, input_seq_len: int, output_seq_len: int):
        """
        Calculates the ranges for horizontal (y) and vertical (h) inputs based on the number of the diagonal.

        Args:
            diagonal_num: the number of the diagonal, in range [0, input_seq_len + output_seq_len - 1)
            input_seq_len: the length of the input sequences (# of tokens) in the current batch
            output_seq_len: the length of the output sequences (# of tokens) in the current batch

        Returns:
            a tuple of two tuples:
                input_range: the range of vertical input values (h) to consider for the current diagonal
                output_range: the range of horizontal output values (y) to consider for the current diagonal

            the two ranges always have the same length, which is between 1 and min(input_seq_len, output_seq_len)
        """
        min_len = min(input_seq_len, output_seq_len)
        max_len = max(input_seq_len, output_seq_len)
        assert 0 <= diagonal_num < min_len + max_len

        if diagonal_num < min_len:
            max_range = (0, diagonal_num + 1)
            min_range = max_range
        elif diagonal_num < max_len:
            max_range = (diagonal_num - (min_len - 1), diagonal_num + 1)
            min_range = (0, min_len)
        else:
            max_range = (diagonal_num - (min_len - 1), max_len)
            min_range = (diagonal_num - (max_len - 1), min_len)

        assert (max_range[1] - max_range[0]) == (min_range[1] - min_range[0])
        assert max_len >= max_range[1] > max_range[0] >= 0
        assert min_len >= min_range[1] > min_range[0] >= 0

        # determine which one is for the input and which one for the output
        if min_len == input_seq_len:        # the input (vertical) is shorter or of equal length to the output
            input_range = min_range
            output_range = max_range
        else:                               # the output (horizontal) is shorter or of equal length to the input
            input_range = max_range
            output_range = min_range

        return input_range, output_range

    @staticmethod
    def __calculate_state_ranges(input_range: Tuple[int, int], output_range: Tuple[int, int]):
        """
        Calculates the indexing ranges for the current diagonal, based on the input and output ranges:
        Args:
            input_range: a tuple of two values (min, max) that represents the range of values taken for the input at
                the current diagonal
            output_range: a tuple of two values (min, max) that represents the range of values taken for the (previous)
                output at the current diagonal

        Returns: three tuples (x_list, y_list) of two integer lists each:
            - ver_ranges: the x and y coordinates for the vertical previous states
            - hor_ranges: the x and y coordinates for the horizontal previous states
            - diag_ranges: the x and y coordinates for the current diagonal (to store the new states correctly)
        """
        # helper function
        def autorange(minmax: Tuple[int, int]) -> List[int]:
            """
            Returns a list of integer indices that represent the given range. If min > max, the reversed range from max+1 to
                min+1 is returned
            :param minmax: the range tuple (min, max) for indices to consider
            :return: an integer list of indices
            """
            min, max = minmax
            if min > max:
                return list(reversed(range(max + 1, min + 1)))
            return list(range(min, max))

        ver_from, ver_to = input_range
        hor_from, hor_to = output_range

        # vertical range
        ver_x_range = (ver_from + 1, ver_to + 1)
        ver_y_range = (hor_to - 1, hor_from - 1)
        ver_ranges = (autorange(ver_x_range), autorange(ver_y_range))

        # horizontal range
        hor_x_range = input_range
        hor_y_range = (hor_to, hor_from)
        hor_ranges = (autorange(hor_x_range), autorange(hor_y_range))

        # indices of the current diagonal
        diag_x_range = ver_x_range
        diag_y_range = hor_y_range
        diag_ranges = (autorange(diag_x_range), autorange(diag_y_range))

        return ver_ranges, hor_ranges, diag_ranges




