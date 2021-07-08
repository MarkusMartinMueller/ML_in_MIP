from aneurysm_utils import evaluation
import pickle 
from aneurysm_utils import postprocessing
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from pathlib import Path
import os 
# postprocess_dict={ 
#     "dbscan":True,
#     "remove_border_candidates":True,
#     "resample":True,
#     "evaluate_dbscan":True,
#     #"invidual_aneurysm_labels":invidual_labels_test,
#     "size":100
                      
# }

#mri_imgs= postprocessing.postprocess(env,aneurysm_labels_new,postprocess_dict)
with open("postprocessed_imgs.pkl", "rb") as fp:   
    mri_imgs= pickle.load(fp)
with open("labels_test.pkl","rb") as fp:
    labels_test= pickle.load(fp)
with open("labels_test_invidual.pkl","rb") as fp:
    invidual_labels_test=pickle.load(fp)
with open("neuralnet_output.pkl","rb") as fp:
    aneurysm_labels_new=pickle.load(fp)

for image in invidual_labels_test:
    print(np.unique(image))
bounding_boxes =postprocessing.bounding_boxes([invidual_labels_test[3]])
evaluation.draw_bounding_box(bounding_boxes[3]["candidates"])

# cases=["A130_R","A118","A120","A115","A133","A073","A072","A084","A077"]
# postprocessing.create_task_one_json(bounding_boxes,cases=cases,path="../../cada-challenge-master/cada_detection/test/reference.json")
# #evaluation.draw_bounding_box(bounding_boxes[1]["candidates"],invidual_labels_test[1],aneurysm_array= mri_imgs[1])
# #plt.show()
# #score_dict2=evaluation.calc_scores_task_2(aneurysm_labels_new,labels_test,mri_imgs,invidual_labels_test)
# #score_dict1=evaluation.calc_scores_task_1(invidual_labels_test,invidual_labels_test,bounding_boxes)
# #print(score_dict1)
# def create_nifits(mri_images,cases,path_cada="../../cada-challenge-master/cada_segmentation/test-gt/",path_datasets='../../datasets/'):
#     data_path = Path(path_datasets)
#     for count,image in enumerate(mri_images):
#         affine = nib.load(data_path / f'{cases[count]}_orig.nii.gz').affine
#         img = nib.Nifti1Image(image, affine)
#         img.to_filename(os.path.join(path_cada,f'{cases[count]}_labeledMasks.nii.gz'))