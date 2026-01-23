
import numpy as np
import torch
import pandas as pd
from jklib.genome import locus

def one_hot_encode(seq):
    map = np.asarray([[0, 0, 0, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

    seq = seq.upper().replace('A', '\x01').replace('C', '\x02')
    seq = seq.replace('G', '\x03').replace('T', '\x04').replace('N', '\x00')
    return map[np.fromstring(seq, np.int8) % 5]

def one_hot_encode_torch(seq):
    """
    DNA 시퀀스를 one-hot 인코딩하여 (1, 4, sequence_length) 형태의 torch.Tensor를 반환합니다.
    A -> [1, 0, 0, 0]
    C -> [0, 1, 0, 0]
    G -> [0, 0, 1, 0]
    T -> [0, 0, 0, 1]
    알 수 없는 문자는 모두 0 벡터로 처리합니다.
    """
    import torch
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq = seq.upper()
    seq_len = len(seq)
    one_hot = np.zeros((seq_len, 4), dtype=np.float32)
    for i, nucleotide in enumerate(seq):
        if nucleotide in mapping:
            one_hot[i, mapping[nucleotide]] = 1.0
    one_hot = one_hot.T  # (4, sequence_length)
    one_hot = np.expand_dims(one_hot, axis=0)  # (1, 4, sequence_length)
    return torch.from_numpy(one_hot)

def replace_dna_ref_to_alt2(dna_sequence, pos, ref, alt, max_distance):
    original_dna_sequence = dna_sequence
    # Note: 5000 offset seems hardcoded in original logic, kept it but this might need adjustment if max_distance logic changes
    # But checking original code: dna_sequence was fetched with start_index=pos-max_distance-max_distance
    # So if max_distance=5000, start_index = pos - 10000.
    # dna_sequence[:5000+max_distance] means up to pos if we consider... wait.
    # In original __init__.py: 
    # start_index=pos-max_distance-max_distance
    # end_index = int(loc.chrEnd)+max_distance+max_distance
    # So the sequence length is roughly 4*max_distance.
    # pos is at index (pos - start_index) = 2*max_distance.
    # In original code line 144: dna_sequence[:5000+max_distance]
    # If max_distance=5000, index is 10000. Correct.
    # If max_distance=1000, index is 6000? But pos is at 2000. 
    # This implies replace_dna_ref_to_alt2 assumes max_distance=5000 logic for the first 5000 offset?
    # Actually, let's keep it as is, but be aware.
    altered_dna_sequence = dna_sequence[:5000+max_distance] + alt + dna_sequence[5000+max_distance+len(ref):]
    return original_dna_sequence, altered_dna_sequence

def custom_dataframe(df_acceptor_loss, df_donor_loss, df_acceptor_gain, df_donor_gain):
    delta_common = pd.concat([df_acceptor_loss, df_donor_loss, df_acceptor_gain, df_donor_gain], axis=1)
    delta_common.index = delta_common.index + 1
    return delta_common
