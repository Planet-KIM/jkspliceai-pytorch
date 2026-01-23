##### System Congiure #####
import threading
import traceback
import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import tensorflow as tf
import tabix

#from multiprocessing import Pool

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
##### Model Configure #####
from keras.models import load_model
import numpy as np
from pandas import DataFrame
import pandas as pd
import h5py


##### Custom Configure #####
from jkspliceai.spliceAI.utils import one_hot_encode, replace_dna_ref_to_alt2, custom_dataframe, load_models
from jklib.genome import locus
from jklib.bioDB import _bioDB

def predict_with_model(model, sequence):
    return model.predict(sequence)


def predict_model(model, encoded_dna_sequence, index, result_array):
    result_array[index] = model.predict(encoded_dna_sequence)

def predicted_values(sequence, strand, model, max_distance, allele):
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
    encoded_dna_sequence = one_hot_encode(sequence)[None, :]
    #if strand == '-':
    #    encoded_dna_sequence = encoded_dna_sequence[:, ::-1, ::-1]
    
    #predicted_value = np.mean([model[m].predict(encoded_dna_sequence) for m in range(5)], axis=0)
    
    num_models = 5
    results = [None] * num_models

    # Create an array to store the threads
    threads = []

    # Create and start each thread
    for m in range(num_models):
        thread = threading.Thread(target=predict_model, args=(model[m], encoded_dna_sequence, m, results))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Calculate the mean of the results
    predicted_value = np.mean(results, axis=0)
    
    if strand == '-':
        predicted_value = predicted_value[:, ::-1, ::1]

    predicted_value_for_common=np.concatenate((predicted_value[0][:max_distance],predicted_value[0][-max_distance:]))

    acceptor_prob_for_common = predicted_value_for_common[ :, 1]
    donor_prob_for_common = predicted_value_for_common[ :, 2]

    
    predicted_value_for_raw=predicted_value[0][max_distance:max_distance+len(allele)]

    acceptor_prob_for_raw = predicted_value_for_raw[ :, 1]
    donor_prob_for_raw = predicted_value_for_raw[ :, 2]
    
    
    return predicted_value, acceptor_prob_for_common, donor_prob_for_common, acceptor_prob_for_raw, donor_prob_for_raw

def delta_values(probs, post_value=False):
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
    delta_prob = prob1 - prob2
    
    #prob1_sorted_index = np.flip((prob1).argsort())
    #prob1_sorted_value = prob1[prob1_sorted_index]

    #prob2_sorted_index = np.flip((prob2).argsort())
    #prob2_sorted_value = prob2[prob2_sorted_index]
    
    delta_sorted_index = np.flip((delta_prob).argsort())
    delta_sorted_value = delta_prob[delta_sorted_index]
    #print(delta_sorted_index, "delta")
    #print(prob1, "prob1")
    #print(prob2, "prob2")
    if post_value:
        prob1_sorted_value = prob1[delta_sorted_index]
        #print(prob1_sorted_value)
        if np.array(prob2).shape[0] == 1:
            prob2_sorted_index = 0
            prob2_sorted_value = prob2[prob2_sorted_index]
        else:
            prob2_sorted_value = prob2[delta_sorted_index]
    else:
        prob1_sorted_value, prob2_sorted_value = [0], [0]

    #print(prob1_sorted_index, prob2_sorted_index, delta_sorted_index)
    return delta_prob, delta_sorted_index, delta_sorted_value, prob1_sorted_value, prob2_sorted_value


def filter_by_value(value_common, index_common, value_raw, index_raw, max_distance, view_threshold, post):
    pre_common, pre_raw, post_common, post_raw = post
    value_list=[]
    index_list=[]
    post_list=[]

    index_for_common = 0
    index_for_raw = 0
    
    while True:
        if len(value_list)==view_threshold:
            break
        
        if index_for_common==value_common.shape[0]:
            #pre_common_value = -10.0
            #post_common_value = -10.0
            candidate_for_common = -10.0
        else:
            #pre_common_value = pre_common[index_for_common]
            #post_common_value = post_common[index_for_common]
            candidate_for_common = value_common[index_for_common]
        #print(index_for_raw, value_raw.shape[0], "test")  
        if index_for_raw==value_raw.shape[0]:
            #pre_raw_value = -10.0
            #post_raw_value = -10.0
            candidate_for_raw = -10.0
        else:
            #print(pre_raw, post_raw)
            #pre_raw_value = pre_raw[index_for_raw]
            #post_raw_value = post_raw[index_for_raw]
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
            index_for_common+=1
            #post_list.append(f"{round(float(pre_common_value), 3)}→{round(float(post_common_value),3)}")
        else:
            value_list.append(candidate_for_raw)
            index_list.append(str(index_raw[index_for_raw])+' (in alt)')
            index_for_raw+=1
            #post_list.append(f"{round(float(pre_raw_value),3)}→{round(float(post_raw_value),3)}")
        
    while len(value_list)<view_threshold:
        value_list.append('None')
        index_list.append('None')
        #post_list.append('None')
    
    return (value_list,index_list, post_list)



def view_result(return_above, max_distance, view_threshold, post_value=False):
    (acceptor_prob_for_common1, donor_prob_for_common1, acceptor_prob_for_raw1, donor_prob_for_raw1,acceptor_prob_for_common2, donor_prob_for_common2, acceptor_prob_for_raw2, donor_prob_for_raw2 ) = return_above
    
    dframe = pd.DataFrame()
    #print("acceptor",acceptor_prob_for_common1, acceptor_prob_for_raw1, len(acceptor_prob_for_common1))
    #print("donor", donor_prob_for_common1, donor_prob_for_raw1, len(donor_prob_for_common1))
    
    acceptor_loss_prob_common, acceptor_loss_sorted_index_common, acceptor_loss_sorted_value_common, acceptor_loss_prob_for_common1, acceptor_loss_prob_for_common2 = delta_values((acceptor_prob_for_common1,acceptor_prob_for_common2))
    acceptor_loss_prob_raw, acceptor_loss_sorted_index_raw, acceptor_loss_sorted_value_raw, acceptor_loss_prob_for_raw1, acceptor_loss_prob_for_raw2  = delta_values((acceptor_prob_for_raw1,acceptor_prob_for_raw2))
    acceptor_loss_value_list, acceptor_loss_index_list, acceptor_loss_post_list = filter_by_value(acceptor_loss_sorted_value_common, acceptor_loss_sorted_index_common, acceptor_loss_sorted_value_raw, acceptor_loss_sorted_index_raw, max_distance, view_threshold,
            post=(acceptor_loss_prob_for_common1, acceptor_loss_prob_for_raw1, acceptor_loss_prob_for_common2, acceptor_loss_prob_for_raw2))
    
    donor_loss_prob_common, donor_loss_sorted_index_common, donor_loss_sorted_value_common, donor_loss_prob_for_common1, donor_loss_prob_for_common2 = delta_values((donor_prob_for_common1,donor_prob_for_common2))
    donor_loss_prob_raw, donor_loss_sorted_index_raw, donor_loss_sorted_value_raw, donor_loss_prob_for_raw1, donor_loss_prob_for_raw2 = delta_values((donor_prob_for_raw1,donor_prob_for_raw2))
    donor_loss_value_list, donor_loss_index_list, donor_loss_post_list = filter_by_value(donor_loss_sorted_value_common, donor_loss_sorted_index_common, donor_loss_sorted_value_raw, donor_loss_sorted_index_raw, max_distance, view_threshold,
            post=(donor_loss_prob_for_common1, donor_loss_prob_for_raw1, donor_loss_prob_for_common2, donor_loss_prob_for_raw2))

    acceptor_gain_prob_common, acceptor_gain_sorted_index_common, acceptor_gain_sorted_value_common, acceptor_gain_prob_for_common2, acceptor_gain_prob_for_common1 = delta_values((acceptor_prob_for_common2,acceptor_prob_for_common1))
    acceptor_gain_prob_raw, acceptor_gain_sorted_index_raw, acceptor_gain_sorted_value_raw, acceptor_gain_prob_for_raw2, acceptor_gain_prob_for_raw1 = delta_values((acceptor_prob_for_raw2,acceptor_prob_for_raw1))
    acceptor_gain_value_list, acceptor_gain_index_list, acceptor_gain_post_list  = filter_by_value(acceptor_gain_sorted_value_common, acceptor_gain_sorted_index_common, acceptor_gain_sorted_value_raw, acceptor_gain_sorted_index_raw, max_distance, view_threshold,
            post=(acceptor_gain_prob_for_common2, acceptor_gain_prob_for_raw2, acceptor_gain_prob_for_common1, acceptor_gain_prob_for_raw1))
    
    donor_gain_prob_common, donor_gain_sorted_index_common, donor_gain_sorted_value_common, donor_gain_prob_for_common2, donor_gain_prob_for_common1 = delta_values((donor_prob_for_common2,donor_prob_for_common1))
    donor_gain_prob_raw, donor_gain_sorted_index_raw, donor_gain_sorted_value_raw, donor_gain_prob_for_raw2, donor_gain_prob_for_raw1 = delta_values((donor_prob_for_raw2,donor_prob_for_raw1))
    donor_gain_value_list, donor_gain_index_list, donor_gain_post_list = filter_by_value(donor_gain_sorted_value_common, donor_gain_sorted_index_common, donor_gain_sorted_value_raw, donor_gain_sorted_index_raw, max_distance, view_threshold,
            post=(donor_gain_prob_for_common2, donor_gain_prob_for_raw2, donor_gain_prob_for_common1, donor_gain_prob_for_raw1))
    
    
    dframe[f'delta score (Acceptor Loss)'] = acceptor_loss_value_list
    dframe[f'pre-mRNA position (Acceptor Loss)'] = acceptor_loss_index_list
    
    dframe[f'delta score (Donor Loss)'] = donor_loss_value_list
    dframe[f'pre-mRNA position (Donor Loss)'] = donor_loss_index_list
    
    dframe[f'delta score (Acceptor Gain)'] = acceptor_gain_value_list
    dframe[f'pre-mRNA position (Acceptor Gain)'] = acceptor_gain_index_list
    
    dframe[f'delta score (Donor Gain)'] = donor_gain_value_list
    dframe[f'pre-mRNA position (Donor Gain)'] = donor_gain_index_list
    if post_value:
        dframe[f'pre score (Acceptor Loss)'] = acceptor_loss_post_list
        dframe[f'pre score (Donor Loss)'] = donor_loss_post_list
        dframe[f'pre score (Acceptor Gain)'] = acceptor_gain_post_list
        dframe[f'pre score (Donor Gain)'] = donor_gain_post_list
        dframe["delta score (Acceptor Loss)"] = dframe["delta score (Acceptor Loss)"].astype(str) + " (" +dframe["pre score (Acceptor Loss)"]+")"
        dframe["delta score (Donor Loss)"] = dframe["delta score (Donor Loss)"].astype(str) + " (" +dframe["pre score (Donor Loss)"]+")"
        dframe["delta score (Acceptor Gain)"] = dframe["delta score (Acceptor Gain)"].astype(str) + " (" +dframe["pre score (Acceptor Gain)"]+")"
        dframe["delta score (Donor Gain)"] = dframe["delta score (Donor Gain)"].astype(str) + " (" +dframe["pre score (Donor Gain)"]+")"
        dframe = dframe.drop(list(dframe.columns)[8:], axis='columns')
    
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

        delta_acceptor_prob, delta_acceptor_sorted_index, delta_acceptor_sorted_value,_,_ = delta_values(probs=(acceptor_prob1, acceptor_prob2))
        delta_donor_prob, delta_donor_sorted_index, delta_donor_sorted_value,_,_ = delta_values(probs=(donor_prob1, donor_prob2))

        df_acceptor_loss,df_donor_loss,df_acceptor_gain, df_donor_gain, df_raw_acceptor_loss,df_raw_donor_loss,df_raw_acceptor_gain,df_raw_donor_gain = view_result((delta_acceptor_prob, delta_donor_prob, delta_acceptor_sorted_index, delta_donor_sorted_index, delta_acceptor_sorted_value, delta_donor_sorted_value, predicted_value_for_raw1, predicted_value_for_raw2),view_threshold,max_distance)

        delta_common = custom_dataframe(df_acceptor_loss, df_donor_loss, df_acceptor_gain, df_donor_gain)
        delta_raw_common = custom_dataframe(df_raw_acceptor_loss, df_raw_donor_loss, df_raw_acceptor_gain, df_raw_donor_gain)
        return { 'delta_common' : delta_common,
                 'delta_raw_common' : delta_raw_common }

    except Exception as e:
        print(e)
        return e


def spliceAI(loc, ref=None, alt=None, max_distance=5000, model='ensemble', view=5, assembly='hg38', verbose=True, todict=False, selector='model', post_value=False):
    """
    spliceAI selector

    Parameters
    ----------
    
    Returns
    -------
    
    """
    if selector=='model':
        common = spliceAI_model(loc=loc, ref=ref, alt=alt, max_distance=max_distance, model=model, view=view, assembly=assembly, verbose=verbose, todict=todict, post_value=post_value)
    elif selector=='indexing':
        common = spliceAI_tabix(loc=loc, ref=ref, alt=alt, assembly=assembly, todict=todict)
    return common

def spliceAI_model(loc, ref=None, alt=None, max_distance=5000, model='ensemble', view=5, assembly='hg38', verbose=True, todict=False, post_value=False):
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
        if 'ensemble' in model:
            model=load_models(model)
        pos = int(loc.chrSta) # 108236168
        start_index=pos-5000-max_distance
        end_index = int(loc.chrEnd)+5000+max_distance
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

        input_dna_sequence1,input_dna_sequence2=replace_dna_ref_to_alt2(dna_sequence,pos,ref,alt,max_distance)

        predicted_value1, acceptor_prob_for_common1, donor_prob_for_common1, acceptor_prob_for_raw1, donor_prob_for_raw1  = predicted_values(input_dna_sequence1, loc.strand, model, max_distance, ref)
        predicted_value2, acceptor_prob_for_common2, donor_prob_for_common2, acceptor_prob_for_raw2, donor_prob_for_raw2  = predicted_values(input_dna_sequence2, loc.strand, model, max_distance, alt)
        
        common_and_raw = view_result((acceptor_prob_for_common1, donor_prob_for_common1, acceptor_prob_for_raw1, donor_prob_for_raw1,acceptor_prob_for_common2, donor_prob_for_common2, acceptor_prob_for_raw2, donor_prob_for_raw2 ),max_distance,view, post_value)
        
        common_and_raw = common_and_raw.round(decimals=2)
        if verbose == False:
            common_and_raw = max_sort_dataframe(common_and_raw, common_and_raw.columns[4])
        if todict == True:
            common_and_raw = common_and_raw.fillna(0)
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
