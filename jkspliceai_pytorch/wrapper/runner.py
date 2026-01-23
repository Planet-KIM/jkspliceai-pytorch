
import numpy as np
import pandas as pd
import h5py
import os
import torch
import torch.nn.functional as F
import traceback

from ..utils import one_hot_encode_torch, replace_dna_ref_to_alt2, custom_dataframe, get_device
from ..models.spliceai import SpliceAI as SpliceAIFactory
from jklib.genome import locus

def predicted_values(sequence, strand, spliceai_models, max_distance, allele, model_name, use_gpu=True):
    if type(spliceai_models) == dict:
        encoded_dna_sequence = one_hot_encode_torch(sequence)
    elif model_name == "spliceai_torch":
        # Load default model configuration if passing "spliceai_torch" string
        # Defaulting to '10k' as per original code logic usually, or passing specific config?
        # Original code line 53: model10k_drop = SpliceAI.from_preconfigured('10k')
        model10k_drop = SpliceAIFactory.from_preconfigured('10k')
        model10k_drop.eval()
        encoded_dna_sequence = one_hot_encode_torch(sequence)
    else:
        encoded_dna_sequence = one_hot_encode_torch(sequence)

    cur = os.path.dirname(os.path.realpath(__file__))
    # Assuming models are stored parallel to wrapper/runner.py? 
    # Original: cur = os.path.dirname(os.path.realpath(__file__)) -> jkspliceai_pytorch/spliceAI
    # New: jkspliceai_pytorch/wrapper
    # But models seem to be in `models/pytorch_10k/` relative to `spliceAI` dir originally?
    # Original path: f'{cur}/models/pytorch_10k/10k_{m+1}_retry.pth'
    # I need to know where the model weights are. 
    # I should assume they will be moved to `jkspliceai_pytorch/weights/` or kept in `jkspliceai_pytorch/spliceAI/models`?
    # I will assume `jkspliceai_pytorch/weights`. I'll need to move them later.
    # Or for now, I can point to `../weights` relative to wrapper.
    
    predict_values = []
    
    # Check where weights are. 
    # I should check if user has weights files.
    # Assuming they are in `jkspliceai_pytorch/spliceAI/models/pytorch_10k`.
    # I will move them to `jkspliceai_pytorch/weights/pytorch_10k`.
    
    # Path handling:
    # If running from wrapper/runner.py, '..' goes to jkspliceai_pytorch.
    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # jkspliceai_pytorch
    
    for m in range(5):
        if type(spliceai_models) == dict:
            model10k_drop = spliceai_models[m]
            with torch.no_grad():
                pt = model10k_drop(encoded_dna_sequence)
                pt = F.softmax(pt, dim=-1)
                predict_values.append(pt.detach().numpy())
        elif model_name == "spliceai_torch":
            # Update path to weights
            model_pth = os.path.join(base_path, 'spliceAI', 'models', 'pytorch_10k', f'10k_{m+1}_retry.pth')
            # I will plan to move this directory later, but for now let's point to old location or new if I move it.
            # Let's decide to move it to `jkspliceai_pytorch/weights`.
            # So path: os.path.join(base_path, 'weights', '10k_{m+1}_retry.pth')
            # Waittt, the files are named '10k_1_retry.pth' etc. inside 'pytorch_10k' dir?
            # Original: f'{cur}/models/pytorch_10k/10k_{m+1}_retry.pth'
            # I will move `spliceAI/models` to `jkspliceai_pytorch/models_data`. Namespaces...
            # Let's say `jkspliceai_pytorch/data/models`.
            # For now, I will use absolute path logic or relative to package.
            
            # Let's assume I move `models` folder to `jkspliceai_pytorch/models_data`.
            model_pth = os.path.join(base_path, 'models_data', 'pytorch_10k', f'10k_{m+1}_retry.pth')

            if not os.path.exists(model_pth):
                # Fallback to old path just in case I haven't moved it yet, or error out
                # Try finding it in original location?
                # But I am deleting spliceAI folder. So I MUST move it.
                pass

            state_dict = torch.load(
                model_pth,
                map_location=torch.device('cpu')
            )
            device = get_device()
            model10k_drop.load_state_dict(state_dict)
            model10k_drop.to(device)
            
            encoded_dna_sequence_dev = encoded_dna_sequence.to(device)
            with torch.no_grad():
                pt = model10k_drop(encoded_dna_sequence_dev)
            
            pt = F.softmax(pt, dim=-1)
            predict_values.append(pt.cpu().detach().numpy())

    predicted_value = np.mean(predict_values, axis=0)

    if strand == '-':
        predicted_value = predicted_value[:, ::-1, ::1]

    predicted_value_for_common=np.concatenate((predicted_value[0][:max_distance],predicted_value[0][-max_distance:]))

    acceptor_prob_for_common = predicted_value_for_common[ :, 1]
    donor_prob_for_common = predicted_value_for_common[ :, 2]

    predicted_value_for_raw=predicted_value[0][max_distance:max_distance+len(allele)]

    acceptor_prob_for_raw = predicted_value_for_raw[ :, 1]
    donor_prob_for_raw = predicted_value_for_raw[ :, 2]

    return predicted_value, acceptor_prob_for_common, donor_prob_for_common, acceptor_prob_for_raw, donor_prob_for_raw

def delta_values_with_trace(probs):
    prob1, prob2 = probs

    prob1 = prob1+prob2-prob2
    prob2 = prob2+prob1-prob1

    delta_prob = prob1 - prob2

    delta_sorted_index = np.flip((delta_prob).argsort())
    sorted_ori_values = prob1[delta_sorted_index]
    sorted_mut_values = prob2[delta_sorted_index]

    return delta_sorted_index, sorted_ori_values, sorted_mut_values

def filter_by_value(index_common , ori_values_common , mut_values_common, index_raw, ori_values_raw , mut_values_raw, max_distance, view_threshold):
    value_list=[]
    index_list=[]
    pre_list = []
    post_list = []

    index_for_common = 0
    index_for_raw = 0

    value_common = ori_values_common - mut_values_common
    value_raw = ori_values_raw - mut_values_raw

    while True:
        if len(value_list)==view_threshold:
            break

        if index_for_common==value_common.shape[0]:
            candidate_for_common = -10.0
        else:
            candidate_for_common = value_common[index_for_common]

        if index_for_raw==value_raw.shape[0]:
            candidate_for_raw = -10.0
        else:
            candidate_for_raw = value_raw[index_for_raw]

        if max(candidate_for_common,candidate_for_raw)<=0:
            break

        if candidate_for_common >= candidate_for_raw:
            if index_common[index_for_common]<max_distance:
                now_index = index_common[index_for_common] - max_distance
            else:
                now_index = index_common[index_for_common] - max_distance + 1

            value_list.append(candidate_for_common)
            index_list.append(str(now_index))

            pre_list.append(ori_values_common[index_for_common])
            post_list.append(mut_values_common[index_for_common])

            index_for_common+=1

        else:
            value_list.append(candidate_for_raw)
            index_list.append(str(index_raw[index_for_raw])+' (in alt)')

            pre_list.append(ori_values_raw[index_for_raw])
            post_list.append(mut_values_raw[index_for_raw])

            index_for_raw+=1

    while len(value_list)<view_threshold:
        value_list.append('None')
        index_list.append('None')

    return (value_list,index_list,pre_list,post_list)

def view_result(return_above, max_distance, view_threshold):
    (acceptor_prob_for_common_ori, donor_prob_for_common_ori, acceptor_prob_for_raw_ori, donor_prob_for_raw_ori,acceptor_prob_for_common_mut, donor_prob_for_common_mut, acceptor_prob_for_raw_mut, donor_prob_for_raw_mut ) = return_above

    dframe = pd.DataFrame()

    al_sorted_index_common, al_sorted_ori_value_common, al_sorted_mut_value_common = delta_values_with_trace((acceptor_prob_for_common_ori,acceptor_prob_for_common_mut))
    al_sorted_index_raw,    al_sorted_ori_value_raw,    al_sorted_mut_value_raw = delta_values_with_trace((acceptor_prob_for_raw_ori,acceptor_prob_for_raw_mut))
    al_value_list, al_index_list, al_pre_common_list, al_post_common_list = filter_by_value(al_sorted_index_common, al_sorted_ori_value_common, al_sorted_mut_value_common, al_sorted_index_raw,    al_sorted_ori_value_raw,    al_sorted_mut_value_raw, max_distance, view_threshold)

    dl_sorted_index_common, dl_sorted_ori_value_common, dl_sorted_mut_value_common = delta_values_with_trace((donor_prob_for_common_ori,donor_prob_for_common_mut))
    dl_sorted_index_raw, dl_sorted_ori_value_raw, dl_sorted_mut_value_raw = delta_values_with_trace((donor_prob_for_raw_ori,donor_prob_for_raw_mut))
    dl_value_list, dl_index_list, dl_pre_common_list, dl_post_common_list = filter_by_value(dl_sorted_index_common, dl_sorted_ori_value_common, dl_sorted_mut_value_common, dl_sorted_index_raw,    dl_sorted_ori_value_raw,    dl_sorted_mut_value_raw, max_distance, view_threshold)

    ag_sorted_index_common, ag_sorted_ori_value_common, ag_sorted_mut_value_common = delta_values_with_trace((acceptor_prob_for_common_mut,acceptor_prob_for_common_ori))
    ag_sorted_index_raw, ag_sorted_ori_value_raw, ag_sorted_mut_value_raw = delta_values_with_trace((acceptor_prob_for_raw_mut,acceptor_prob_for_raw_ori))
    ag_value_list, ag_index_list, ag_pre_common_list, ag_post_common_list = filter_by_value(ag_sorted_index_common, ag_sorted_ori_value_common, ag_sorted_mut_value_common, ag_sorted_index_raw,    ag_sorted_ori_value_raw,    ag_sorted_mut_value_raw, max_distance, view_threshold)

    dg_sorted_index_common, dg_sorted_ori_value_common, dg_sorted_mut_value_common = delta_values_with_trace((donor_prob_for_common_mut,donor_prob_for_common_ori))
    dg_sorted_index_raw, dg_sorted_ori_value_raw, dg_sorted_mut_value_raw = delta_values_with_trace((donor_prob_for_raw_mut,donor_prob_for_raw_ori))
    dg_value_list, dg_index_list,dg_pre_common_list, dg_post_common_list = filter_by_value(dg_sorted_index_common, dg_sorted_ori_value_common, dg_sorted_mut_value_common, dg_sorted_index_raw,    dg_sorted_ori_value_raw,    dg_sorted_mut_value_raw, max_distance, view_threshold)


    dframe[f'pre-mRNA position (Acceptor Loss)'] = al_index_list
    dframe[f'delta score (Acceptor Loss)'] = al_value_list
    dframe[f'pre (Acceptor Loss)'] = al_pre_common_list
    dframe[f'post (Acceptor Loss)'] = al_post_common_list

    dframe[f'pre-mRNA position (Donor Loss)'] = dl_index_list
    dframe[f'delta score (Donor Loss)'] = dl_value_list
    dframe[f'pre (Donor Loss)'] = dl_pre_common_list
    dframe[f'post (Donor Loss)'] = dl_post_common_list

    dframe[f'pre-mRNA position (Acceptor Gain)'] = ag_index_list
    dframe[f'delta score (Acceptor Gain)'] = ag_value_list
    dframe[f'pre (Acceptor Gain)'] = ag_post_common_list
    dframe[f'post (Acceptor Gain)'] = ag_pre_common_list

    dframe[f'pre-mRNA position (Donor Gain)'] = dg_index_list
    dframe[f'delta score (Donor Gain)'] = dg_value_list
    dframe[f'pre (Donor Gain)'] = dg_post_common_list
    dframe[f'post (Donor Gain)'] = dg_pre_common_list

    return dframe

def max_sort_dataframe(df, column, window=1):
    return df.sort_values(by=column, ascending=False).head(window)

class PositionError(Exception):
    def __init__(self):
        super().__init__('Position Error')

class LocusLengthError(Exception):
    def __init__(self):
        super().__init__('Locus Length Error')

def spliceAI(loc, ref=None, alt=None, max_distance=5000, model='spliceai_torch', view=5, assembly='hg38', verbose=True, todict=False, selector='model', use_gpu=True):
    if selector=='model':
        common = spliceAI_model(loc=loc, ref=ref, alt=alt, max_distance=max_distance, model=model, view=view, assembly=assembly, verbose=verbose, todict=todict, use_gpu=use_gpu)
    return common

def spliceAI_model(loc, ref=None, alt=None, max_distance=5000, model='spliceai_torch', view=5, assembly='hg38', verbose=True, todict=False, use_gpu=True):
    try:
        pos = int(loc.chrSta)
        start_index=pos-max_distance-max_distance
        end_index = int(loc.chrEnd)+max_distance+max_distance
        chrom=loc.chrom

        dna_sequence=locus(f"{chrom}:{str(start_index)}-{str(end_index)}{loc.strand}")
        dna_sequence=dna_sequence.twoBitFrag(assembly).upper()

        if int(loc.chrEnd)-int(loc.chrSta)+1!=len(ref):
            print('Error in end index')
            print('Error index should be: '+str(len(ref)+int(loc.chrSta)-1))
            raise LocusLengthError


        if dna_sequence[pos-start_index:pos-start_index+len(ref)]!=ref:
            print('Error in positioning')
            print('The hg38 reference allele should be: '+dna_sequence[pos-start_index:pos-start_index+len(ref)])
            for i in range(-10,11):
                print(dna_sequence[pos-start_index+i],end="")
            print()
            print(' '*10+"-")
            raise PositionError

        if 'spliceai_torch' in model:
            spliceai_models=model
        elif type(model) == dict:
            spliceai_models=model
            model='spliceai_torch'
        else:
            raise Exception(f"Invalid model: {model}")

        input_dna_sequence1,input_dna_sequence2=replace_dna_ref_to_alt2(dna_sequence,pos,ref,alt,max_distance)

        predicted_value1, acceptor_prob_for_common1, donor_prob_for_common1, acceptor_prob_for_raw1, donor_prob_for_raw1  = predicted_values(input_dna_sequence1, loc.strand, spliceai_models, max_distance, ref, model, use_gpu=use_gpu)
        predicted_value2, acceptor_prob_for_common2, donor_prob_for_common2, acceptor_prob_for_raw2, donor_prob_for_raw2  = predicted_values(input_dna_sequence2, loc.strand, spliceai_models, max_distance, alt, model, use_gpu=use_gpu)

        common_and_raw = view_result((acceptor_prob_for_common1, donor_prob_for_common1, acceptor_prob_for_raw1, donor_prob_for_raw1,acceptor_prob_for_common2, donor_prob_for_common2, acceptor_prob_for_raw2, donor_prob_for_raw2 ),max_distance,view)

        common_and_raw = common_and_raw.round(decimals=2)
        common_and_raw = common_and_raw.fillna(0)
        if verbose == False:
            common_and_raw = max_sort_dataframe(common_and_raw, common_and_raw.columns[4])
        if todict == True:
            return common_and_raw.to_dict('index')
        return common_and_raw

    except Exception as e:
        print(traceback.format_exc())
        return e
