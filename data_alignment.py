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
import numpy as np
from mmsdk import mmdatasdk as md
from subprocess import check_call, CalledProcessError

from mmsdk.mmdatasdk import log, computational_sequence
import numpy
import time
import struct
from tqdm import tqdm_notebook as tqdm

import pickle, os, json, codecs

# create folders for storing the data
if not os.path.exists(DATA_PATH):
    check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)
    

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

    output_byte = b''
    

    for datapoint in features:

        y = list(map(lambda x: struct.pack('>f', x), datapoint))
        byte_datapoint = b''.join(y)

        output_byte += byte_datapoint

    with open(file_name + ext, 'wb') as file:
        file.write(header + output_byte)


def save_intervals(intervals, feature_type, folder, video_number):
    #set default extension
    ext = '.json'
    video_number = 'intervals_' + video_number
    file_name = os.path.join(folder, feature_type, video_number)
    
    if type(intervals) != list:
        intervals = intervals.tolist()
    json.dump(intervals, codecs.open(file_name + ext, 'w', encoding='utf-8'), indent=4)
    

class new_mmdataset(md.mmdataset):
    #TODO: Need tqdm bar for this as well
    def get_relevant_entries(self,reference):
        relevant_entries={}
        relevant_entries_np={}

        #pbar = tqdm(total=count,unit=" Computational Sequence Entries",leave=False)

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
                relev_intervals_np=numpy.concatenate(relevant_entries[otherseq_key][key]["intervals"],axis=0)                                 
                relev_features_np=numpy.concatenate(relevant_entries[otherseq_key][key]["features"],axis=0)
                sorted_indices=sorted(range(relev_intervals_np.shape[0]),key=lambda x: relev_intervals_np[x,0])                               
                relev_intervals_np=relev_intervals_np[sorted_indices,:]                         
                relev_features_np=relev_features_np[sorted_indices,:]

                relevant_entries_np[otherseq_key][key]={}
                relevant_entries_np[otherseq_key][key]["intervals"]=relev_intervals_np
                relevant_entries_np[otherseq_key][key]["features"]=relev_features_np
            log.status("Pre-alignment done for <%s> ..."%otherseq_key)
        return relevant_entries_np
    
    def intersect_and_copy_upsampling(self, ref_all, relevant_entry, epsilon, log_file, feature_info):
        #ref_all: reference_entries[other_key][entry_key]['intervals'] e.g. [COVAREP][some video id]['intervals']
        #relevant_entry: relevant_entries[other_key][entry_key] e.g. [COVAREP][some video id]
        #epsilon: error allowed in alignment
        #ref_time < one interval in relevant_entry
        
        pbar_small=log.progress_bar(total=ref_all.shape[0],unit=" Segments",leave=False)
        pbar_small.set_description("Aligning: " + feature_info)
        
        sub=relevant_entry["intervals"]
        features=relevant_entry["features"]
        
        #finding where intersect happens
        pointer_b = 0 # for relevant_entry
        aligned_sub = []
        aligned_feature = []
        
        for i, inter in enumerate(ref_all):
            #print(pointer_b)
            if (abs(inter[0]-inter[1])<epsilon):
                pbar_small.update(1)
                continue
            pointer_c = pointer_b
            while(pointer_c < sub.shape[0]):
                if (inter[0] - sub[pointer_c][0]) > (-epsilon) and (sub[pointer_c][1] - inter[0]) > (-epsilon):
                    aligned_sub.append(sub[pointer_c])
                    aligned_feature.append(features[pointer_c])
                    break
                else:
                    pointer_c += 1
                
            if pointer_c == sub.shape[0]:
                diff = list(map(lambda x: abs(inter[0] - x), sub[:, 0]))
                min_diff = min(diff)
                pointer_c = diff.index(min_diff)
                with open(log_file, 'w+') as fi:
                    fi.write('no corresponding frame, find the closest one, {}, difference: {}\n'.format(feature_info, min_diff))
                aligned_sub.append(sub[pointer_c])
                aligned_feature.append(features[pointer_c])
            else:
                pointer_b = pointer_c

            pbar_small.update(1)
        
        aligned_sub = np.array(aligned_sub)
        aligned_feature = np.array(aligned_feature)
        zero_idx = np.where(np.isinf(aligned_feature))
        aligned_feature[zero_idx] = 0

        pbar_small.close()
        return aligned_sub,aligned_feature
    
    def align_upsampling_and_save(self, reference, id_idx, collapse_function=None, epsilon = 10e-6):
        folder = '/data/mifs_scratch/yw454/cmumosei_aligned'
        log_file = './mosei_alignment_log.txt'
        #aligned_output = {}
        count = 0
        
        ##self.computational_sequences.keys are COVERAP, OpenFace, WordVec, etc
        #for sequence_name in self.computational_sequences.keys():
        #    #init a dictionary to store different featues seperately
        #    aligned_output[sequence_name]={}
        
        if reference not in self.computational_sequences.keys():
            log.error("Computational sequence <%s> does not exist in dataset"%reference,error=True)
        
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
            
            if entry_key in id_idx:
                stored_idx = id_idx.index(entry_key)
                #if stored_idx < 104 or (stored_idx > 104 and stored_idx < 1781):
                if stored_idx < 1781 or stored_idx == 1815:
                    continue
            
            all_intersects = {}
            all_intersect_features = {}
            
            #for sequence_name in self.computational_sequences.keys():
            #    all_intersects[sequence_name] = []
            #    all_intersect_features[sequence_name] = []
            
            ref_all=refseq[entry_key]['intervals']
                
            #aligning all sequences to ref sequence (previous: align refer to refer as well, now: not include refer)
            #otherseq_key: other features; entry_key: some video id
                
            for otherseq_key in list(self.computational_sequences.keys()):
                if otherseq_key != reference:
                    feature_info = 'reference: {}, other feature {}, video id: {}'.format(reference, otherseq_key, entry_key)
                    intersects,intersects_features=self.intersect_and_copy_upsampling(ref_all,relevant_entries[otherseq_key][entry_key],epsilon, log_file, feature_info)
                else:
                    intersects,intersects_features=refseq[entry_key]['intervals'][:,:],refseq[entry_key]['features'][:,:]
                    
                #print(type(intersects[0]))
                #print(type(intersects_features[0]))
                #print(len(intersects[0]))
                #print(len(intersects_features[0]))
                all_intersects[otherseq_key] = intersects
                all_intersect_features[otherseq_key] = intersects_features
                    

            #save features per video
            for sequence_name in self.computational_sequences.keys():
                video_code = id_idx.index(entry_key)
                video_code = str(video_code).zfill(6)
                
                save_htk_format(all_intersect_features[sequence_name], sequence_name, folder, video_code)
                save_intervals(all_intersects[sequence_name], sequence_name, folder, video_code)
                print('alignment saved for video {} feature {}.'.format(video_code, sequence_name))
            
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

basic_dataset.align_upsampling_and_save('COVAREP', video_ids)
