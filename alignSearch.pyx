import numpy as np
cimport numpy as np

def cython_fill_table(np.ndarray[np.float32_t, ndim=2] table, np.ndarray[np.float32_t, ndim=2] lpz, np.ndarray[np.int_t, ndim=2] ground_truth, np.ndarray[np.int_t, ndim=1] offsets, np.ndarray[np.int_t, ndim=1] utt_begin_indices, int blank, float argskip_prob):
    cdef int c
    cdef int t
    cdef int offset = 0
    cdef float mean_offset
    cdef int offset_sum = 0
    cdef int lower_offset
    cdef int higher_offset
    cdef float switch_prob, stay_prob, skip_prob
    cdef float prob_max = -1000000000
    cdef float lastMax 
    cdef int lastArgMax 
    cdef int next_utt_index = 1
    cdef int utt_offset = 0
    cdef np.ndarray[np.int_t, ndim=1] cur_offset = np.zeros([ground_truth.shape[1]], np.int) - 1
    cdef float max_lpz_prob 
    cdef float p 
    cdef int s

    mean_offset = (lpz.shape[0] - table.shape[0]) / float(table.shape[1])
    print("Mean offset: " + str(mean_offset))
    lower_offset = int(mean_offset)
    higher_offset = lower_offset + 1

    table[0, 0] = 0
    for c in range(table.shape[1]):   

        if c > 0:
            #offset = higher_offset if offset_sum / float(c) < mean_offset else lower_offset
            offset = min(max(0, lastArgMax - table.shape[0] // 2), min(higher_offset, (lpz.shape[0] - table.shape[0]) - offset_sum))
            #print(offset, offset_sum, lpz.shape[0], table.shape[0], lpz.shape[0] - table.shape[0], lastArgMax - table.shape[0] // 2)
            #print(c, lastArgMax, lastMax)

            for s in range(ground_truth.shape[1] - 1):
                cur_offset[s + 1] = cur_offset[s] + offset
            cur_offset[0] = offset

            offset_sum += offset
        offsets[c] = offset_sum
        lastArgMax = -1
        lastMax = 0

        if c == utt_begin_indices[next_utt_index]:
            utt_offset = offset_sum - offsets[utt_begin_indices[next_utt_index - 1]]

        for t in range((1 if c == 0 else 0), table.shape[0]):
            switch_prob = prob_max
            max_lpz_prob = prob_max
            for s in range(ground_truth.shape[1]):
                if ground_truth[c, s] != -1:
                    if t >= table.shape[0] - (cur_offset[s] - 1) or t - 1 + cur_offset[s] < 0 or c == 0:
                        p = prob_max
                    else:
                        p = table[t - 1 + cur_offset[s], c - (s + 1)] + lpz[t + offset_sum, ground_truth[c, s]]
                    switch_prob = max(switch_prob, p)

                    max_lpz_prob = max(max_lpz_prob, lpz[t + offset_sum, ground_truth[c, s]])

            #skip_prob = prob_max if t >= table.shape[0] - offset else (table[t + offset, c - 1] + argskip_prob)
            if t - 1 < 0:
                stay_prob = prob_max 
            elif c == 0:
                stay_prob = 0
            else:
                stay_prob = table[t - 1, c] + max(lpz[t + offset_sum, blank], max_lpz_prob)

            table[t, c] = max(switch_prob, stay_prob)
            #if c == utt_begin_indices[next_utt_index] and t + utt_offset < table.shape[0]:
            #    table[t, c] = max(table[t, c], table[t + utt_offset, utt_begin_indices[next_utt_index - 1]] + argskip_prob * (utt_begin_indices[next_utt_index] - utt_begin_indices[next_utt_index - 1]))
                
            if lastArgMax == -1 or lastMax < table[t, c]:
                lastMax = table[t, c]
                lastArgMax = t

        if c == utt_begin_indices[next_utt_index]:
            next_utt_index += 1
            if next_utt_index >= utt_begin_indices.shape[0]:
                next_utt_index = 0

    c = table.shape[1] - 1
    t = table[:, c].argmax()
    return t, c


def cython_fill_chapter_end_table(np.ndarray[np.float32_t, ndim=2] table, np.ndarray[np.float32_t, ndim=2] lpz, np.ndarray[np.int_t, ndim=1] ground_truth, np.ndarray[np.int_t, ndim=2] non_space_chars, int blank):
    cdef int c
    cdef int t

    table[0, 0] = 0
    for c in range(table.shape[1]):
        for t in range(table.shape[0]):
            if c == 0:
                table[t, c] = table[t - 1, c] + lpz[t, blank]
                non_space_chars[t, c] = non_space_chars[t - 1, c]
            elif t == 0:
                if lpz[t, blank] > lpz[t, ground_truth[c - 1]]:
                    table[t, c] = lpz[t, blank]
                    non_space_chars[t, c] = 0
                else:
                    table[t, c] = lpz[t, ground_truth[c - 1]]
                    non_space_chars[t, c] = 1
            else:
                if table[t - 1, c] + lpz[t, blank] > table[t - 1, c - 1] + lpz[t, ground_truth[c - 1]]:
                    table[t, c] = table[t - 1, c] + lpz[t, blank]
                    non_space_chars[t, c] = non_space_chars[t - 1, c]
                else:
                    table[t, c] = table[t - 1, c - 1] + lpz[t, ground_truth[c - 1]]
                    non_space_chars[t, c] = non_space_chars[t - 1, c - 1] + 1

    