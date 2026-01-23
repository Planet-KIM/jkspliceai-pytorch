##### System Congiure #####
import threading
import traceback
import os
import sys
import tabix

#from multiprocessing import Pool

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
##### Model Configure #####
import numpy as np
from pandas import DataFrame
import pandas as pd
import h5py

##### Custom Configure #####
from jkspliceai.spliceAI.utils import one_hot_encode, one_hot_encode_torch, replace_dna_ref_to_alt2, custom_dataframe, load_models
from jklib.genome import locus
from jklib.bioDB import _bioDB

import torch
import torch.nn.functional as F
from spliceai_pytorch import SpliceAI   

def predicted_values(sequence, strand, spliceai_models, max_distance, allele, model_name):
    """
    Parameters
    ----------
    sequence: str
        DNA sequence
    strand: str
        '-' or '+'
    model: str
        splice model
    allele: str
        'ref' or 'alt'

    Returns
    -------
    predicted_value:
    predicted_value_for_common:
    predicted_value_for_raw:
    acceptor_prob:
    donor_prob:

    """
    if type(spliceai_models) == dict:
        encoded_dna_sequence = one_hot_encode_torch(sequence)
    elif model_name == "spliceai_torch":
        model10k_drop = SpliceAI.from_preconfigured('10k_drop')
        model10k_drop.eval()  # 평가 모드로 전환
        encoded_dna_sequence = one_hot_encode_torch(sequence)
    else:
        encoded_dna_sequence = one_hot_encode(sequence)[None, :]
    #print(encoded_dna_sequence.shape)

    predict_values = []
    for m in range(5):
        if type(spliceai_models) == dict:
            model10k_drop = spliceai_models[m]
            with torch.no_grad():
                pt = model10k_drop(encoded_dna_sequence)
                pt = F.softmax(pt, dim=-1)
                predict_values.append(pt.detach().numpy())
        elif model_name== "spliceai_torch":
            cur = os.path.dirname(os.path.realpath(__file__))
            model10k_drop.load_state_dict(torch.load(f'{cur}/models/pytorch/10k_{m+1}_drop_retry.pth', weights_only=True))
            #model10k_drop.load_state_dict(torch.load(f'{cur}/models/pytorch_10k/10k_{m+1}_retry.pth', weights_only=True))
            #model10k_drop.eval()  # 평가 모드로 전환
            pt = model10k_drop(encoded_dna_sequence)
            pt = F.softmax(pt, dim=-1)
            predict_values.append(pt.detach().numpy())
        else:
            pt = spliceai_models[m].predict(encoded_dna_sequence)
            predict_values.append(pt)

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





def delta_values_with_trace(probs):  ########## YM : 불필요한 인자 전달 제거
    """
    Parameters
    ----------
    probs: tuple
        for example,
        probs tuple value is (acceptor_prob1, acceptor_prob2)

    Returns
    -------
    delta_prob:
    delta_sorted_index:
    delta_sorted_value:

    """
    prob1, prob2 = probs
    
    prob1 = prob1+prob2-prob2 # to match the shapes of prob1 and prob2
    prob2 = prob2+prob1-prob1 # to match the shapes of prob1 and prob2
    
    delta_prob = prob1 - prob2

    delta_sorted_index = np.flip((delta_prob).argsort())
#     delta_sorted_value = delta_prob[delta_sorted_index]
    sorted_ori_values = prob1[delta_sorted_index]
    sorted_mut_values = prob2[delta_sorted_index]
    

    return delta_sorted_index, sorted_ori_values, sorted_mut_values


def filter_by_value(index_common , ori_values_common , mut_values_common, index_raw, ori_values_raw , mut_values_raw, max_distance, view_threshold):
    ########## YM : 불필요한 인자 전달 제거 , index 고려하기 위해 인자에 index 관련 argument 추가
    
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
            
            ########## YM : pre_list는 이전값 (original seq로 부터 얻어진 값), post_list는 mut로부터 얻어진 값을 저장
            
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
    
    
    ########## YM : 인자 정리, 데이터프레임에 list 추가
    
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

###################YC_E##################



def spliceAI_dataportal(chrom,pos,ref,alt,strand,max_distance,model='ensemble',view_threshold=5,assembly='hg38'):
    """
    Parameters
    ----------
    chrom: int
    pos: int
    ref: str
    alt: str
    strand: str
    max_distance: str
    model:
        default value is spliceai model
    view_threshold: int
        default value is 5
    assembly: str
        default value is 'hg38'
        or 'hg19' <- not yet

    Returns
    -------
    delta_common: pandas.dataframe
        json formatting
    delta_raw_common: pandas.dataframe
        json formatting

    """
    try:
        if model == 'ensemble':
            model=load_models(model)
        start_index= pos-5000-max_distance
        end_index= pos+5000+max_distance
        chrom='chr'+str(chrom)

        if strand =='-':
            start_index=start_index

        dna_sequence=locus(f"{chrom}:{str(start_index)}-{str(end_index)}+")
        dna_sequence=dna_sequence.twoBitFrag2(assembly).upper()

        if dna_sequence[pos-start_index:pos-start_index+len(ref)]!=ref:
            print('The hg38 reference allele should be: '+dna_sequence[pos-start_index:pos-start_index+len(ref)])
            for i in range(-10,11):
                print(dna_sequence[pos-start_index+i],end="")
            print()
            print(' '*10+"-")
            raise PositionError

        input_dna_sequence1,input_dna_sequence2=replace_dna_ref_to_alt2(dna_sequence,pos,ref,alt,max_distance)

        predicted_value1, predicted_value_for_common1, predicted_value_for_raw1, acceptor_prob1, donor_prob1  = predicted_values(input_dna_sequence1, strand, model, max_distance, ref)
        predicted_value2, predicted_value_for_common2, predicted_value_for_raw2, acceptor_prob2, donor_prob2  = predicted_values(input_dna_sequence2, strand, model, max_distance, alt)

        delta_acceptor_prob, delta_acceptor_sorted_index, delta_acceptor_sorted_value = delta_values(probs=(acceptor_prob1, acceptor_prob2))
        delta_donor_prob, delta_donor_sorted_index, delta_donor_sorted_value = delta_values(probs=(donor_prob1, donor_prob2))

        df_acceptor_loss,df_donor_loss,df_acceptor_gain, df_donor_gain, df_raw_acceptor_loss,df_raw_donor_loss,df_raw_acceptor_gain,df_raw_donor_gain = view_result((delta_acceptor_prob, delta_donor_prob, delta_acceptor_sorted_index, delta_donor_sorted_index, delta_acceptor_sorted_value, delta_donor_sorted_value, predicted_value_for_raw1, predicted_value_for_raw2),view_threshold,max_distance)

        delta_common = custom_dataframe(df_acceptor_loss, df_donor_loss, df_acceptor_gain, df_donor_gain)
        delta_raw_common = custom_dataframe(df_raw_acceptor_loss, df_raw_donor_loss, df_raw_acceptor_gain, df_raw_donor_gain)
        return { 'delta_common' : delta_common,
                 'delta_raw_common' : delta_raw_common }

    except Exception as e:
        print(e)
        return e


def spliceAI(loc, ref=None, alt=None, max_distance=5000, model='ensemble', view=5, assembly='hg38', verbose=True, todict=False, selector='model'):
    """
    spliceAI selector

    Parameters
    ----------
    
    Returns
    -------
    
    """
    if selector=='model':
        common = spliceAI_model(loc=loc, ref=ref, alt=alt, max_distance=max_distance, model=model, view=view, assembly=assembly, verbose=verbose, todict=todict)
    elif selector=='indexing':
        common = spliceAI_tabix(loc=loc, ref=ref, alt=alt, assembly=assembly, todict=todict)
    return common

def spliceAI_model(loc, ref=None, alt=None, max_distance=5000, model='ensemble', view=5, assembly='hg38', verbose=True, todict=False):
    """
    Parameters
    ----------
    loc: locus
        accoding to ucsu coordinate
    ref: str
        default value is None
    alt: str
        default value is None
    max_distance: str
        default value is None
    model:
        default value is spliceai model
    view: int
        default value is 5
    assembly: str
        default value is 'hg38'
        or 'hg19' <- not yet

    Returns
    -------
    delta_common: pandas.DataFrame
        json formatting
    delta_raw_common: pandas.DataFrame
        json formatting

    Examples
    --------
    spliceAI_model(locus('chr11:108236168-108236168+'),ref='AG',alt='G',max_distance=10000,view=100)
    
    
    splice = spliceAI_server(locus('chr11:108279637-108279637+'), ref='TATC', alt='T', max_distance=1000, view=10)
    splice['delta_common']
    splice['delta_raw_common']

    splice = spliceAI_server(locus('chr6:64081870-64081870-'), ref='C', alt='T', strand='-', max_distance=1000, view=5)
    splice['delta_common']
    splice['delta_raw_common']
    """
    try:
        pos = int(loc.chrSta) # 108236168
        start_index=pos-max_distance-max_distance
        end_index = int(loc.chrEnd)+max_distance+max_distance
        chrom=loc.chrom

        if loc.strand =='-':
            start_index=start_index

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

        if 'ensemble' in model:
            spliceai_models=load_models(model=model, max_distance=max_distance)
        elif 'spliceai_torch' in model:
            spliceai_models=model
        elif type(model) == dict:
            spliceai_models=model
            model='spliceai_torch'
        input_dna_sequence1,input_dna_sequence2=replace_dna_ref_to_alt2(dna_sequence,pos,ref,alt,max_distance)

        predicted_value1, acceptor_prob_for_common1, donor_prob_for_common1, acceptor_prob_for_raw1, donor_prob_for_raw1  = predicted_values(input_dna_sequence1, loc.strand, spliceai_models, max_distance, ref, model)
        predicted_value2, acceptor_prob_for_common2, donor_prob_for_common2, acceptor_prob_for_raw2, donor_prob_for_raw2  = predicted_values(input_dna_sequence2, loc.strand, spliceai_models, max_distance, alt, model)
        
        common_and_raw = view_result((acceptor_prob_for_common1, donor_prob_for_common1, acceptor_prob_for_raw1, donor_prob_for_raw1,acceptor_prob_for_common2, donor_prob_for_common2, acceptor_prob_for_raw2, donor_prob_for_raw2 ),max_distance,view)
        
        common_and_raw = common_and_raw.round(decimals=2)
        common_and_raw = common_and_raw.fillna(0)
        if verbose == False:
            common_and_raw = max_sort_dataframe(common_and_raw, common_and_raw.columns[4])
        if todict == True:
            #common_and_raw = common_and_raw.fillna(0)
            return common_and_raw.to_dict('index')
        return common_and_raw

    except Exception as e:
        print(traceback.format_exc())
        return e


# todict 설정
def spliceAI_tabix(loc, ref=None, alt=None, assembly='hg38', todict=False):

    if type(loc) == str:
        locusStr=loc
        pass
    else:
        locusStr=loc.toString()

    result = []
    for r in spliceAI_raw(locusStr,assembly):

        chrN = r[0]
        pos = int(r[1])

        ref1 = r[2]
        alt1 = r[3]

        if ref==ref1 and alt==alt1:
            pass
        elif ref==None and alt==None:
            pass
        else:
            continue

        tL = r[4].split('|')
        geneName = tL[1]

        score = dict(zip(('AG','AL','DG','DL'),list(map(float,tL[2:6]))))
        relPos = dict(zip(('AG','AL','DG','DL'),list(map(int,tL[6:]))))

        result.append({'chrN':chrN,'pos':pos,'ref':ref1,'alt':alt1,'geneName':geneName,'score':score,'relPos':relPos})
    
    result = list({item['alt']: item for item in result}.values())
    if todict == False:
        result = pd.DataFrame(result)
    return result


def spliceAI_raw(locusStr, assembly='hg38'):

    if locusStr[:3] == 'chr':
        locusStr = locusStr[3:]

    tb_spliceai_snv = tabix.open(f'/commons/Reference/Tool_reference/SpliceAI/genome_scores_v1.3/spliceai_scores.raw.snv.{assembly}.vcf.gz')
    tb_spliceai_indel = tabix.open(f'/commons/Reference/Tool_reference/SpliceAI/genome_scores_v1.3/spliceai_scores.raw.indel.{assembly}.vcf.gz')

    snv_records = tb_spliceai_snv.querys(locusStr)
    indel_records = tb_spliceai_snv.querys(locusStr)

    resultL = []

    for result in snv_records:
        resultL.append((result[0],result[1],result[3],result[4],result[7]))

    for result in indel_records:
        resultL.append((result[0],result[1],result[3],result[4],result[7]))

    return resultL

"""
class SpliceAI(_bioDB):
    def __init__(self, db='spliceai', version=None):
        super().__init__(db, )

        if version:
            self.db = eval(f"self.db.{version}")
        else:
            self.db = eval("self.db.default")

    def query(self, loc, ref, alt, assembly='hg38', verbose=False):

        locusStr = super()._filter_loc(loc.toString())[3:]
        snv, indel = (self.db.snv, self.db.indel)

        tb_spliceai_snv = tabix.open(snv)
        tb_spliceai_indel = tabix.open(indel)

        snv_records = tb_spliceai_snv.querys(locusStr)
        indel_records = tb_spliceai_snv.querys(locusStr)

        resultL = []

        for result in snv_records:
            resultL.append((result[0],result[1],result[3],result[4],result[7]))

        for result in indel_records:
            resultL.append((result[0],result[1],result[3],result[4],result[7]))
        
        res = []
        for r in resultL:

            tL = r[4].split('|')

            geneName = tL[1]
            score = dict(zip(('AG','AL','DG','DL'),list(map(float,tL[2:6]))))
            relPos = dict(zip(('AG','AL','DG','DL'),list(map(int,tL[6:]))))

            res.append({'chrN' : r[0],
                        'pos' : int(r[1]),
                        'ref' : r[2],
                        'alt' : r[3],
                        'geneName' : geneName,
                        'score':score,
                        'relPos':relPos})

        return res
"""

def max_sort_dataframe(df, column, window=1):
    """
    Parameters
    ----------
    df = pandas.Dataframe
        delta_common and delta_raw_common dataframe
    window: int
        user selection window

    Returns
    -------
    df: pandas.DataFrame
        sorted dataframe
    """
    return df.sort_values(by=column, ascending=False).head(window)

class PositionError(Exception):
    def __init__(self):
        super().__init__('Position Error')

class LocusLengthError(Exception):
    def __init__(self):
        super().__init__('Locus Length Error')

