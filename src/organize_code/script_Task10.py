
# Folder fnames
import src.utils as ut
import utils_xml as utx
import os
DIR= '../annotation_pascal'
xml_fname = '../datasets/AICity_data/train/S03/c010/Anotation_test.xml'
xml_fname2 = '../week1_results/Anotation_G01.xml'
fname='../datasets/AICity_data/train/S03/c010/gt/gt.txt'

BboxList =utx.get_bboxes_from_aicity(xml_fname2)
#BboxList = ut.get_bboxes_from_MOTChallenge(fname)
print(type(BboxList))
print(BboxList)
#ut.folderPascal2xml(xml_fname,xml_fname2,DIR)
#file_list = []
#i = 0
#for file_name in os.listdir(DIR):
#    if file_name.endswith(".xml"):
#        file_list.append(os.path.join(DIR,file_name))
#        i +=1

#print file_list
#bboxes,pd_bboxes = ut.get_bboxes_from_pascal(file_list, 2)
#print('BBOX- DICT?')
#print(bboxes)
#print('BBOX- PD')
#print(pd_bboxes)
#xml_fname = '/home/noamor/Documents/repo/m6/datasets/AICity_data/train/S03/c010/Anotation_test.xml'
#xml_fname2 = '/home/noamor/Documents/repo/m6/datasets/AICity_data/train/S03/c010/Anotation_test3.xml'
#ut.add_track2xml(xml_fname,xml_fname2,pd_bboxes)

#ut.create_aicity_xml_file('trial_name.xml', pd_bboxes)
#boxes,pd.DataFrame(bboxes)get_bboxes_from_aicity(fnames)
