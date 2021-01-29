from constants import SDK_PATH, DATA_PATH, WORD_EMB_PATH, CACHE_PATH
import sys

if SDK_PATH is None:
    print("SDK path is not specified! Please specify first in constants/paths.py")
    exit(0)
else:
    sys.path.append(SDK_PATH)

import mmsdk
import os
import re
import json
import numpy as np
    
import time
import math
import struct
from tqdm import tqdm

from mmsdk import mmdatasdk as md
from mmsdk.mmdatasdk import log, computational_sequence
from subprocess import check_call, CalledProcessError

# create folders for storing the data
if not os.path.exists(DATA_PATH):
    check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)

DATASET = md.cmu_mosei
ALL_VIDEO = DATASET.standard_folds.standard_train_fold + \
     DATASET.standard_folds.standard_test_fold + \
     DATASET.standard_folds.standard_valid_fold

def save_htk_format(features, feature_type, folder, video_number):
    '''
    This function works for function align_upsampling_and_save. It save feature vectors of one video id into one 
    htk format file. 
    feature: feature of one video
    All files are of USER type.
    number of samples being consistent with number of samples in csd file
    sample bytes for COVAREP: 4 * 74 = 296
    sample period: 10 000.0 us (100000)
    paramkind: USER (9)
    '''
    #set default extension
    ext = '.txt'
    file_name = os.path.join(folder, feature_type, video_number)

    num_sample = len(features)
    byte_n_sample = num_sample.to_bytes(4, byteorder='big')

    period = 100000
    byte_period = period.to_bytes(4, byteorder = 'big')

    if feature_type == 'COVAREP':
        sample_b = 296
        ext = '.cov'
    elif feature_type == 'WordVec':
        sample_b = 1200
        ext = '.wvec'

    byte_sample_b = sample_b.to_bytes(2, byteorder = 'big')

    sample_type = 9
    byte_sample_type = sample_type.to_bytes(2, byteorder = 'big')
    header = byte_n_sample + byte_period + byte_sample_b + byte_sample_type
    
    #get rid of inf
       
    inf_idx = np.where(np.isinf(features))
    features[inf_idx] = 0

    output_byte = b''

    for datapoint in features:
        y = list(map(lambda x: struct.pack('>f', x), datapoint))
        byte_datapoint = b''.join(y)

        output_byte += byte_datapoint

    with open(file_name + ext, 'wb') as file:
        file.write(header + output_byte)


class new_mmdataset(md.mmdataset):
    def get_relevant_entries(self,reference):
        '''
        loading all data in the dataset into a dictionary. Did not take more than 2 minutes when running
        '''
        
        
        relevant_entries={}
        relevant_entries_np={}


        #otherseq_key: OpenFace, wordvec, etc
        for otherseq_key in set(list(self.computational_sequences.keys()))-set([reference]):
            relevant_entries[otherseq_key]={}
            relevant_entries_np[otherseq_key]={}
            sub_compseq=self.computational_sequences[otherseq_key]
            # for some_id in all video ids
            for key in list(sub_compseq.data.keys()):
                keystripped=key.split('[')[0]
                if keystripped not in relevant_entries[otherseq_key]:                           
                    relevant_entries[otherseq_key][keystripped]={}
                    relevant_entries[otherseq_key][keystripped]["intervals"]=[]                     
                    relevant_entries[otherseq_key][keystripped]["features"]=[]                                                            
                    
                relev_intervals=self.computational_sequences[otherseq_key].data[key]["intervals"]                                             
                relev_features=self.computational_sequences[otherseq_key].data[key]["features"]         
                if len(relev_intervals.shape)<2:
                    relev_intervals=relev_intervals[None,:]
                    relev_features=relev_features[None,:]

                relevant_entries[otherseq_key][keystripped]["intervals"].append(relev_intervals)
                relevant_entries[otherseq_key][keystripped]["features"].append(relev_features)
                
            for key in list(relevant_entries[otherseq_key].keys()):
                relev_intervals_np=np.concatenate(relevant_entries[otherseq_key][key]["intervals"],axis=0)                                 
                relev_features_np=np.concatenate(relevant_entries[otherseq_key][key]["features"],axis=0)
                sorted_indices=sorted(range(relev_intervals_np.shape[0]),key=lambda x: relev_intervals_np[x,0])                               
                relev_intervals_np=relev_intervals_np[sorted_indices,:]                         
                relev_features_np=relev_features_np[sorted_indices,:]

                relevant_entries_np[otherseq_key][key]={}
                relevant_entries_np[otherseq_key][key]["intervals"]=relev_intervals_np
                relevant_entries_np[otherseq_key][key]["features"]=relev_features_np
            log.status("Pre-alignment done for <%s> ..."%otherseq_key)
        return relevant_entries_np
    
    def upsampling(self, relevant_entry, video_id):
        interval = relevant_entry['intervals']
        feature = relevant_entry['features']
        shape =  relevant_entry['features'].shape[1]
        
        pbar_small=log.progress_bar(total=interval.shape[0],unit=" Segments",leave=False)
        pbar_small.set_description("Aligning: " + video_id)
        
        last_frame = math.floor(np.amax(interval) * 100)
        #first_frame = math.floor(np.amin(interval) * 100)
        #length = max(last_frame, last_frame - first_frame)
        upsampled_feature = np.zeros((last_frame + 1, shape))
        for i, inter in enumerate(interval):
            for idx in range(math.floor(inter[0] * 100), math.floor(inter[1] * 100)):
                upsampled_feature[idx] = feature[i]
            pbar_small.update(1)
        pbar_small.close()
        return upsampled_feature
                
    
    def upsampling_and_save(self, reference, id_idx, collapse_function=None, epsilon = 10e-6):
        folder = '/data/mifs_scratch/yw454/cmumosei_aligned'
        #not_enough_label_file = './mosei_notenough_lable_videos.txt'
        
        ##self.computational_sequences.keys are COVERAP, OpenFace, WordVec, etc
        #for sequence_name in self.computational_sequences.keys():
        #    #init a dictionary to store different featues seperately
        #    aligned_output[sequence_name]={}
        
        if reference not in self.computational_sequences.keys():
            log.error("Computational sequence <%s> does not exist in dataset"%reference,error=True)
        
        modality = list(self.computational_sequences.keys())
        support = ['COVAREP', 'WordVec']
        for m in modality:
            if m not in support:
                raise ValueError('feature type not supported {}'.format(m))
        
        #get data of reference feature
        refseq=self.computational_sequences[reference].data
        #unifying the dataset, removing any entries that are not in the reference computational sequence
        self.unify()
        
        #building the relevant entries to the reference - what we do in this section is simply removing all the [] from the entry ids and populating them into a new dictionary
        log.status("Pre-alignment based on <%s> computational sequence started ..."%reference)
        
        relevant_entries=self.get_relevant_entries(reference)
        log.status("Alignment starting ...")
        
        
        pbar = log.progress_bar(total=len(refseq.keys()),unit=" Computational Sequence Entries",leave=False)
        pbar.set_description("Overall Progress")
        # for some_id in all video ids
        for entry_key in list(refseq.keys()):
            not_enough_label = False
            
            if entry_key not in ALL_VIDEO:
                continue
            
            if entry_key in id_idx:
                stored_idx = id_idx.index(entry_key)
                if stored_idx <= 2132:
                #if stored_idx != 1781:
                    continue
            
            video_code = id_idx.index(entry_key)
            video_code = str(video_code).zfill(6)
            for otherseq_key in list(self.computational_sequences.keys()):
                if otherseq_key == reference:
                    # save reference (COVAREP) data
                    processed_feature = refseq[entry_key]['features'][:, :]
                else:
                    #save upsampled (wordvec) data
                    processed_feature = self.upsampling(relevant_entries[otherseq_key][entry_key], entry_key)

                save_htk_format(processed_feature, otherseq_key, folder, video_code)
                
                print('alignment saved for video {} feature {}.'.format(video_code, otherseq_key))
            
            pbar.update(1)
        pbar.close()


# define your different modalities - refer to the filenames of the CSD files
basic_dict={'COVAREP': DATA_PATH + 'CMU_MOSEI_COVAREP.csd', 
            'WordVec': DATA_PATH + 'CMU_MOSEI_TimestampedWordVectors.csd'}
second_dict = {'Facet': DATA_PATH + 'CMU_MOSEI_VisualFacet42.csd',
            'OpenFace': DATA_PATH + 'CMU_MOSEI_VisualOpenFace2.csd'}
other_dict = {'Word': DATA_PATH + 'CMU_MOSEI_TimestampedWords.csd',
             'Phone': DATA_PATH + 'CMU_MOSEI_TimestampedPhones.csd'}
label_dict = {'mylabels':DATA_PATH + 'CMU_MOSEI_Labels.csd'}


basic_dataset = new_mmdataset(basic_dict)
second_dataset = new_mmdataset(second_dict)
[COVAREP, WordVec] = [basic_dataset.computational_sequences['COVAREP'],
                        basic_dataset.computational_sequences['WordVec']] 
    
video_id_record = os.path.join('/data/mifs_scratch/yw454/cmumosei_aligned', 'video_id.json')

with open(video_id_record, 'r') as json_file:
    video_ids = json.load(json_file)

basic_dataset.upsampling_and_save('COVAREP', video_ids)
           
            
            
            

                    
            
            
            
                    
        
        
        
        
        
        
