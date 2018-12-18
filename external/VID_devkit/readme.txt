=================================================
Introduction
=================================================

This is the documentation of the ILSVRC2015 Competitions Development Kit.

There are two datasets for this year's main competitions and two datasets
for the taster competitions:
* Object detection dataset (DET) -- same as ILSVRC2014
* Classification-localization dataset (CLS-LOC) -- same as ILSVRC2012
* Object detection from video dataset (VID) -- new in ILSVRC2015
* Places2 Scene classification dataset (SCENE) -- new in ILSVRC2015

Table of contents:
  1. Overview of all datasets
    1.1 Images
    1.2 Videos
    1.3 Object categories and types of labels
    1.4 Statistics and data basics
  2. Detection (DET) competition details
    2.1 DET object categories
    2.2 DET images and annotations
    2.3 DET evaluation
    2.4 DET submission format
    2.5 DET evaluation routines
  3. Classification and localization (CLS-LOC) competition details
    3.1 CLS-LOC object categories
    3.2 CLS-LOC images and annotations
    3.3 CLS-LOC submission format
    3.4 CLS-LOC evaluation routines
  4. Object detection from video (VID) competition details
    4.1 VID object categories
    4.2 VID snippets and annotations
    4.3 VID evaluation
    4.4 VID submission format
    4.5 VID evaluation routines
  5. Places2 scene classification (SCENE) competition details

Please contact ilsvrc@image-net.org for questions, comments, or bug reports.


=================================================
1. Overview of all datasets
=================================================

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
1.1 Images
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

For CLS-LOC and DET competition there are three sets of images and
labels: training data, validation data, and test data. There is no
overlap between the three sets. However, many of the images and
labels are shared between the DET and CLS-LOC competitions.

                 Number of images

    Dataset      TRAIN      VALIDATION     TEST
    -------------------------------------------------
    CLS-LOC     1281167       50000       100000
    DET          456567       20121        51294
    -------------------------------------------------
    in both      267770       15522        30901

Key differences between the DET and CLS-LOC images:

- Every image in CLS-LOC dataset are collected using queries for a
  particular synset whereas many DET images are collected using more
  general queries, designed to retrieve scene-like images that might
  contain categories in the competition.

- In order to focus on detection, images from CLS-LOC dataset where
  the CLS-LOC target object is too large (greater than 50% of the
  image area) are not included in the DET validation and test sets.

- The DET training data includes manually verified negative training
  examples while there are no negative training images in CLS-LOC
  dataset

Key difference between ILSVRC2014 and ILSVRC2015 data:

- The DET test dataset is partially refreshed with 11142 new images.
- Layout of both dataset is reorganized to make it easier to use.
- Fix some problematic images and annotations in CLS-LOC dataset.

More details about data collection are available in [1] for CLS-LOC
dataset and [2] for DET dataset.


*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
1.2 Videos
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

For VID competition there are three sets of data and labels:
training data, validation data, and test data. There is no overlap
between the three sets.

                      Number of snippets

    Dataset                  TRAIN    VALIDATION   TEST
    ------------------------------------------------------
    VID                       3862        555       937
    ------------------------------------------------------

- The basic unit for VID is snippet, which contains many frames. A video
  might contain several snippets, which are either manually selected or
  manually verified from snippets detected by shot boundary detector [4],
  so that they are not only research interesting, but also of manageable
  complexity for bounding box detection and labeling.

- Most of the videos are queried using Youtube API [5] with topicIDs
  manually selected from Freebase [6] which are related to a particular
  synset. Queried videos are further manually verified if they contain
  the synset or not. Cartoon and unnatural videos are deleted.

- All visible objects from all frames in each snippet are exhaustively
  annotated with bounding box using VATIC [7]. There are two types of
  boxes: the manually annotated ones and the generated ones using linear
  interpolation. Manually annotated boxes are mostly tight with the visible
  part of the object.

Note:
- Due to the scale of the data, a small subset of frames in each snippets
  may nevertheless have errors in the annotations.


*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
1.3 Object categories and types of labels
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

There are 1000 object categories annotated in CLS-LOC dataset, 200
in DET dataset, and 30 in VID dataset.

- Generally, the object categories in the DET dataset tend to be
  basic-level categories, such as 'dog' and 'bird', whereas CLS-LOC
  contains more fine-grained classes, e.g. different dog breeds.

- The object categories in the VID dataset are those which can move,
  such as 'airplane' and 'dog', whereas CLS-LOC and DET contains more
  static classes, e.g. 'chair' and 'backpack'.

- Each annotated object category corresponds 1-1 to a synset (set of
  synonymous nouns) in WordNet.

- The categories are selected such that there is no overlap between
  synsets: for any pair of synsets i and j in the CLS-LOC dataset, i
  is not an ancestor of j in the WordNet hierarchy. Similarly, this is
  true for any pair of synsets in DET dataset. (This is not, however,
  true between the CLS-LOC and DET datasets: e.g., CLS-LOC contains
  several breeds of dogs whereas DET contains the parent 'dog'
  category.) The categories in VID dataset are selected from the DET
  dataset.

CLS-LOC labels:

- Every image in training, validation and test sets has a single
  image-level label specifying the presence of one object category

- A subset of images in the training set and all of the validation and
  test images have bounding box annotations for the single annotated
  object category

DET labels:

- Every image in the training set has one or more image-level labels,
  for the presence or absence of one or more object categories.
  Bounding boxes are provided around instances of the present
  categories.

- The new DET training images added for ILSVRC2014 have all
  instances of all 200 categories labeled with bounding boxes.

- All instances of all 200 categories are labeled with bounding boxes
  in the DET validation set and in the DET test set.

VID labels:

- Every snippet in the training set has one or more image-level labels.
  Bounding boxes are provided around all visible instances of the present
  categories across all frames.


*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
1.4 Statistics and data basics
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

DET statistics:

  Training:

    - 456567 images and 478807 bounding box annotations around object
      instances.

    - between 461 and 67513 positive images per synset (median 823),
      annotated with the corresponding bounding boxes to yield between
      502 and 74517 object instances per synset (median 1008)

    - between 42945 and 70626 negative training images per synset
      (median 64614.5)

  Validation:

    - 20121 images fully annotated with the 200 object categories,
      yielding 55502 object instances

    - between 31 and 12823 instances per category (median 111)

  Test:

    - 50000 images fully annotated with the 200 object categories.
      (these annotations will not be released)

CLS-LOC statistics:

  Training:

    - 1281167 images, with between 732 and 1300 per synset

    - bounding box annotations for at least 100 (and often many more)
      images from each synset

  Validation:

    - 50000 images, at 50 images per category, with bounding box
      annotations for the target category

  Test:

    - 100000 images, at 100 images per category, with bounding box
      annotations for the target category

VID statistics:

  Training:

    - 3862 snippets fully annotated with the 30 object categories, yielding
      866870 manual bounding boxes for 1122397 frames.

    - between 56 and 458 positive snippets per synset (median 116),

    - between 6 and 5492 frames per snippet (median 180)

  Validation:

    - 555 snippets fully annotated with the 30 object categories, yielding
      135949 manual bounding box annotations for 176126 frames.

    - between 4 and 64 snippets per category (median 17)

    - between 11 and 2898 frames per snippet (median 232)

  Test:

    - 937 snippets fully annotated with the 30 object categories.
      (these annotations will not be released)

Packaging details:

The link for downloading the data can be obtained by registering for
the challenge at

    http://www.image-net.org/challenges/LSVRC/2015

All images and bounding box annotations are packed in a single tar
file for both DET and CLS-LOC dataset. All images are in JPEG format.
All bounding box annotations are in PASCAL VOC format. They can be
parsed using the provided development toolkit. There is one XML
file for each image with bounding box annotations. If the image
filename is X.JPEG, then the bounding box file is named as X.xml.

For more information on the bounding box annotations, visit:

http://www.image-net.org/download-bboxes


=================================================
2. DET competition details
=================================================

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
2.1 DET object categories
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

There are 200 synsets in the DET dataset, and the validation and test
results are evaluated on these synsets.

The 200 synsets in the DET dataset are part of the larger ImageNet
hierarchy. In the training set some object instances have been further
labeled as belonging to a particular subcategory -- for example, some
instances of 'dog' in the training set may actually be associated with
a more specific 'fox terrier' breed label. This is the 'subcategory'
label of the 'object' element in the XML annotation. You are free to
use this information as you see fit during training.

---------------------------------------------------
2.1.1 DET object information in data/meta_det.mat
---------------------------------------------------

Information about the 200 synsets is available in the 'synsets'
array of data/meta_det.mat.

To access this data in Matlab, type

  load data/meta_det.mat;
  synsets

and you will see

synsets =

1x200 struct array with fields:
    ILSVRC2015_DET_ID
    WNID
    name
    description

Each entry in the struct array corresponds to a synset, i, and contains
fields:

'ILSVRC2015_DET_ID' is an integer ID assigned to each synset. All the
synsets used in the ILSVRC2015 detection competition are assigned to an
ID between 1 and 200. The synsets are sorted by ILSVRC2015_DET_ID in
the 'synsets' array, i.e.  synsets(i).ILSVRC2015_DET_ID == i. For
submission of prediction results, ILSVRC2015_DET_ID is used as the
synset labels.

'WNID' is the WordNet ID of a synset. It is used to uniquely identify
a synset in ImageNet or WordNet. It is used as the object name in xml
annotations.


*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
2.2 DET images and annotations
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

---------------------------
2.2.1 DET training data
---------------------------

DET TRAINING LISTS OF IMAGES:

For every synset X of the 200 synsets there are three types of
training images:

- Positive training images: These images contain at least one instance
  of X, and all instances of X in the image are annotated with
  bounding boxes. This is labeled as 1.

- Partial positive training images: These images contain at least one
  instance of X but not all instances of X may be annotated with
  bounding boxes.  Some examples of such images would be pictures of
  bunches of bananas, crowds of people, etc. This is labeled as 0.

- Negative training images: These images do not contain any instances
  of X. They were queried to look similar to the environments that the
  X might be found in, and were manually verified. This is labeled as -1.

(Due to the scale of this data, additional unannotated instances may
occasionally occur. By our estimates this happens on less than 1% of
the positive/negative training images.)

For every one of the 200 synsets there is one corresponding file
listing the training images that fall into each of these categories:

    ImageSets/DET/train_1.txt
    ImageSets/DET/train_2.txt

      ...
    ImageSets/DET/train_200.txt

In these files, every line contains just an image_id and a label.
As described above, positive image is labeled as 1, partial positive
image is labeled as 0, and negative image is labeled as -1. The full
image name is image_id.JPEG, and the corresponding annotations is
image_id.xml. Images without any annotated objects may not have a
corresponding xml file.

DET TRAINING DATA PACKAGING:

Within the DET training images tar archive, there are three types of
of files:

 - For images from ImageNet there is a folder ILSVRC2013_train for each
   synset, named by its WNID. The image files within are named as x_y.JPEG,
   where x is the WNID and y is an integer (not fixed width and not
   necessarily consecutive). The synset ids x are a superset of the
   200 object categories used in this competition.

 - There are images which were queried specifically for the DET
   dataset to serve as negative training data. These images are
   packaged as 11 folders:

    ILSVRC2013_DET_train_extra0
    ILSVRC2013_DET_train_extra1
    ...
    ILSVRC2013_DET_train_extra10

   Each batch contains roughly 10K images. These images are named
   ILSVRC2013_train_z.JPEG, where z is an integer (not necessarily
   consecutive).

- The images collected for ILSVRC2014 are packaged as 7 folders:

    ILSVRC2014_DET_train_0000
    ILSVRC2014_DET_train_0001
    ...
    ILSVRC2014_DET_train_0006

  Each batch contains roughly 10K images. These images are named
  ILSVRC2014_train_z.JPEG, where z is an integer.

-----------------------------
2.2.2 DET validation data
-----------------------------

The validation images are listed in

    ImageSets/DET/val.txt

Every line of the file contains an

   <image_id> <index>

The full image name is image_id.JPEG. The index is used to identify
the image during evaluation. The indices are sorted, so they simply
correspond to line numbers in the file.

Each of these images has been fully annotated with each of the 200
object categories.

Note:
- For each category, there may be a small subset of validation and
  test images which have too much ambiguity in the labels. These
  images are blacklisted (for this category only) and are
  automatically skipped by the evaluation script. Such images from
  the validation set along with the corresponding categories are
  listed in data/ILSVRC2015_det_validation_blacklist.txt.
- Due to the scale of the data, a small subset of validation and
  test images may nevertheless have errors in the annotations.

-----------------------
2.2.3 DET test data
-----------------------

The test images are listed in

    ImageSets/DET/test.txt

This file follows the same format as val.txt above. The test images
may be downloaded as a single archive.  The ground truth annotations
will not be released.


*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
2.3 DET evaluation
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

The detection task is judged as in the PASCAL VOC challenge [3] by the
average precision (AP) on a precision/recall curve. A predicted
bounding box B of class X is considered to have properly localized a
ground truth bounding box Bgt of class X if

  IOU(B,Bgt) = (B intersection Bgt) / (B union Bgt) >= thr

For Bgt of size m*n pixels, we define

                      m*n
  thr = min(0.5, ------------- )
                 (m+10)*(n+10)

Duplicate detections on a single object instance are considered false
detections.  All detections which do not properly localize a ground
truth bounding box of the right class according to this threshold are
considered false detections.

The winner of the detection challenge will be the team which achieves
first place accuracy on the most object categories.

Discussion regarding choice of thr:

    The PASCAL VOC threshold is thr = 0.5. However, for smaller
  objects even deviations of a few pixels would be unacceptable
  according to this threshold.For example, consider an object of size
  10x10 pixels, with a detection window of 20x20 pixels which fully
  contains that object.  This would be an error of ~5 pixels on each
  dimension, which is average human annotation error. However, the IOU
  in this case would be 100/400 = 0.25, far below thr=0.5. Thus for
  smaller objects we loosen the threshold to allow for the annotation
  to extend up to 5 pixels on average in each direction around the
  object. In practice, this changes the threshold on objects which are
  smaller than ~25x25 pixels, and affects ~5.5% of objects on the
  ILSVRC2014 validation set.


*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
2.4 DET submission format
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

Submission of the results will consist of a text file
with one line per predicted object. It looks as follows:

  <image_index> <ILSVRC2015_DET_ID> <confidence> <xmin> <ymin> <xmax> <ymax>

image_index corresponds to the image index from val.txt or test.txt,
depending on which set of images is being annotated.

An example submission file based on the validation data is provided:

  evaluation/demo.val.pred.det.txt


*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
2.5 DET evaluation routines
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

The Matlab routine for evaluating a submission for the detection task is

   evaluation/eval_detection.m

(Note: this function has been significantly optimized since
ILSVRC2013.)

To see an example of using the routine, start Matlab
in the 'evaluation/' folder and type

   demo_eval_det

and you will see something similar to the following output (there will
be additional lines starting with ' eval_detection ::' to document
progress of the evaluation):

DETECTION TASK
pred_file: demo.val.pred.det.txt
meta_file: ../data/meta_det.mat
eval_file: ../../ImageSets/DET/val.txt
blacklist_file: ../data/ILSVRC2015_det_validation_blacklist.txt
NOTE: you can specify a cache filename and the ground truth data will be automatically cached to save loading time in the future.
We already provide the cached ground truth file: ../data/ILSVRC2015_det_ground_truth.mat
Please enter the path to the Validation bounding box annotations directory: ../../Annotations/DET/val
-------------
Category        AP
accordion       0.564
airplane        0.528
ant             0.548
antelope        0.626
apple           0.506
 ... (190 categories)
water bottle    0.521
watercraft      0.646
whale           0.548
wine bottle     0.610
zebra           0.772
 - - - - - - - -
Mean AP:         0.590
Median AP:       0.601


=================================================
3. CLS-LOC competition details
=================================================

The CLS-LOC data is unchanged since ILSVRC 2012.

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
3.1 CLS-LOC object categories
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

The 1000 synsets in the CLS-LOC dataset are part of the larger
ImageNet hierarchy and we can consider the subset of ImageNet
containing these low level synsets.

All information on the synsets is in the 'synsets' array in

  data/meta_clsloc.mat

(This file is the same as in ILSVRC2012/13 devkits except the id field
was renamed to ILSVRC2015_CLSLOC_ID.)

To access this data in Matlab, type

  load data/meta_clsloc.mat;
  synsets

and you will see

   synsets =

   1x1000 struct array with fields:
       ILSVRC2015_CLSLOC_ID
       WNID
       name
       description
       num_train_images

Each entry in the struct array corresponds to a synset, i, and contains
fields:

'ILSVRC2015_CLSLOC_ID' is an integer ID assigned to each synset.
The synsets are sorted by ILSVRC2015_CLSLOC_ID in the 'synsets' array,
i.e. synsets(i).ILSVRC2015_CLSLOC_ID == i. For submission of prediction
results, ILSVRC2015_CLSLOC_ID is used as the synset labels.

'WNID' is the WordNet ID of a synset. It is used to uniquely identify
a synset in ImageNet or WordNet. It is used as the object name in xml
annotations.

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
3.2 CLS-LOC images and annotations
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

---------------------------
3.2.1 CLS-LOC training data
---------------------------

Each image is considered as belonging to a particular synset
X. This image is then guaranteed to contain an X. See [1] for more
details of the collection and labeling strategy.

The CLS-LOC training images may be downloaded as a single tar
archive. Within it there is a tar file for each synset, named by its
WNID. The image files are named as x_y.JPEG, where x is the WNID and y
is an integer (not fixed width and not necessarily consecutive). All
images are in JPEG format.

The bounding box annotations for at least 100 images from each synset
can be downloaded in xml format.

-----------------------------
3.2.2 CLS-LOC validation data
-----------------------------

There are a total of 50,000 validation images. They are named as

      ILSVRC2012_val_00000001.JPEG
      ILSVRC2012_val_00000002.JPEG
      ...
      ILSVRC2012_val_00049999.JPEG
      ILSVRC2012_val_00050000.JPEG

There are 50 validation images for each synset.

The classification ground truth of the validation images is in
    data/ILSVRC2015_clsloc_validation_ground_truth.txt,
where each line contains one ILSVRC2015_ID for one image, in the
ascending alphabetical order of the image file names.

The localization ground truth for the validation images can be downloaded
in xml format.

Notes:
(1) data/ILSVRC2015_clsloc_validation_ground_truth.txt is unchanged
since ILSVRC2012.
(2) As in ILSVRC2012 and 2013, 1762 images (3.5%) in the validation
set are discarded due to unsatisfactory quality of bounding boxes
annotations. The indices to these images are listed in
data/ILSVRC2015_clsloc_validation_blacklist.txt. The evaluation script
automatically excludes these images. A similar percentage of images
are discarded for the test set.

-----------------------
3.2.3 CLS-LOC test data
-----------------------

There are a total of 100,000 test images. The test files are named as

      ILSVRC2012_test_00000001.JPEG
      ILSVRC2012_test_00000002.JPEG
      ...
      ILSVRC2012_test_00099999.JPEG
      ILSVRC2012_test_00100000.JPEG

There are 100 test images for each synset. The ground truth annotations will
not be released.


*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
3.3 CLS-LOC submission format
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

The submission of results on test data will consist of a text file
with one line per image, in the alphabetical order of the image file
names, i.e. from ILSVRC2012_test_00000001.JPEG to
ILSVRC2012_test_0100000.JPEG. Each line contains up to 5 detected
objects, sorted by confidence in descending order. The format is as
follows:

    <label(1)> <xmin(1)> <ymin(1)> <xmax(1)> <ymax(1)> <label(2)> <xmin(2)> <ymin(2)> <xmax(2)> <ymax(2)> ....

The predicted labels are the ILSVRC2015_IDs ( integers between 1 and
1000 ).  The number of labels per line can vary, but not more than 5
(extra labels are ignored).

Example file on the validation data is

  evaluation/demo.val.pred.loc.txt


*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
3.4 CLS-LOC evaluation routines
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

The Matlab routine for evaluating the submission is

./evaluation/eval_clsloc.m

To see an example of using the routines, start Matlab
in the 'evaluation/' folder and type
       demo_eval_clsloc;

and you will see something similar to the following output (there may
be additional lines starting with ' eval_clsloc ::' to document
progress of the evaluation):

CLASSIFICATION WITH LOCALIZATION TASK
pred_file: demo.val.pred.loc.txt
ground_truth_file: ../data/ILSVRC2015_clsloc_validation_ground_truth.txt
blacklist_file: ../data/ILSVRC2015_clsloc_validation_blacklist.txt
Please enter the path to the Validation bounding box annotations directory: ../../Annotations/CLS-LOC/val
# guesses vs clsloc error vs cls-only error
    1.0000    1.0000    0.9992
    2.0000    0.9999    0.9982
    3.0000    0.9998    0.9971
    4.0000    0.9997    0.9962
    5.0000    0.9996    0.9955

In this demo, we take top i ( i=1...5) predictions (and ignore the
rest) from your result file and plot the error as a function of the
number of guesses.

Only the error with 5 guesses will be used to determine the winner.

(The demo.val.pred.loc.txt used here is a synthetic result.)


=================================================
4. VID competition details
=================================================

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
4.1 VID object categories
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

There are 30 synsets in the VID dataset, and the validation and test
results are evaluated on these synsets. The 30 synsets in the VID
dataset are selected from the 200 DET synsets.

---------------------------------------------------
4.1.1 VID object information in data/meta_vid.mat
---------------------------------------------------

Information about the 30 synsets is available in the 'synsets' array
of data/meta_vid.mat.

To access this data in Matlab, type

  load data/meta_vid.mat;
  synsets

and you will see

synsets =

1x30 struct array with fields:
    ILSVRC2015_VID_ID
    WNID
    name
    description

Each entry in the struct array corresponds to a synset, i, and contains
fields:

'ILSVRC2015_VID_ID' is an integer ID assigned to each synset. All the
synsets used in the ILSVRC2015 VID competition are assigned to an ID
between 1 and 30. The synsets are sorted by ILSVRC2015_DET_ID in the
'synsets' array, i.e.  synsets(i).ILSVRC2015_DET_ID == i. For submission
of prediction results, ILSVRC2015_DET_ID is used as the synset labels.

'WNID' is the WordNet ID of a synset. It is used to uniquely identify
a synset in ImageNet or WordNet. It is used as the object name in xml
annotations.


*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
4.2 VID snippets and annotations
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

---------------------------
4.2.1 VID training data
---------------------------

VID TRAINING LISTS OF SNIPPETS:

For every one of the 30 synsets there is one corresponding file listing
the training snippets that contain instances of the category. A snippet
will be listed for multiple categories if it contains instances of each:

    ImageSets/VID/train_1.txt
    ImageSets/VID/train_2.txt

      ...
    ImageSets/VID/train_30.txt

In these files, every line contains just a snippet_id and a label. The
frames in a snippet are stored in a directory as JPEG images with naming
like snippet_id/%06d.JPEG, and the corresponding annotations is
snippet_id/%06d.xml. The frame starts with index 0.

VID TRAINING DATA PACKAGING:

The snippets collected for ILSVRC2015 are packaged as 4 folders:

    ILSVRC2015_VID_train_0000
    ILSVRC2015_VID_train_0001
    ILSVRC2015_VID_train_0002
    ILSVRC2015_VID_train_0003

  Each batch contains at most 1000 snippets. These snippets are named as
  ILSVRC2015_train_z/%06d.JPEG, where z is an integer of format %05d%03d,
  where the first part inidicates the video_id and the second part is the
  snippet_id in the video.

-----------------------------
4.2.2 VID validation data
-----------------------------

The validation snippets are extracted into frames and are listed in

    ImageSets/VID/val.txt

Every line of the file contains an

   <frame_name> <frame_index>

The full frame name is snippet_id/%06d.JPEG. The index is used to
identify the frame during evaluation. The indices are sorted, so they
simply correspond to line numbers in the file.

-----------------------
4.2.3 VID test data
-----------------------

The test snippets are extracted into frames and are listed in

    ImageSets/VID/test.txt

This file follows the same format as val.txt above. The test snippets
may be downloaded as a single archive.  The ground truth annotations
will not be released.


*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
4.3 VID evaluation
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

Only frames manually annotated with each of the 30 object categories
will be evaluated. Frames with generated boudning box are ignored
during evaluation.

The evaluation metric is the same as for the object detection task,
meaning objects which are not annotated will be penalized, as will
be duplicate detections (two annotations for the same object instance).


*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
4.4 VID submission format
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

Submission of the results will consist of a text file
with one line per predicted object. It looks as follows:

  <frame_index> <ILSVRC2015_VID_ID> <confidence> <xmin> <ymin> <xmax> <ymax>

frame_index corresponds to the frame index from val.txt or test.txt,
depending on which set of images is being annotated.

An example submission file based on the validation data is provided:

  evaluation/demo.val.pred.vid.txt


*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
4.5 VID evaluation routines
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

The Matlab routine for evaluating a submission for the detection task is

   evaluation/eval_vid_detection.m

To see an example of using the routine, start Matlab
in the 'evaluation/' folder and type

   demo_eval_vid

and you will see something similar to the following output (there will
be additional lines starting with ' eval_vid_detection ::' to document
progress of the evaluation):

DETECTION FROM VIDEO TASK
pred_file: demo.val.pred.vid.txt
meta_file: ../data/meta_vid.mat
eval_file: ../../ImageSet/VID/val.txt
blacklist_file:
NOTE: you can specify a cache filename and the ground truth data will be automatically cached to save loading time in the future.
We already provide the cached ground truth file: ../data/ILSVRC2015_vid_ground_truth.mat
Please enter the path to the Validation bounding box annotations directory: ../../Annotations/VID/val
-------------
Category        AP
airplane        0.134
antelope        0.085
 ... (25 categories)
watercraft      0.120
whale           0.136
zebra           0.174
 - - - - - - - -
Mean AP:         0.213
Median AP:       0.236


=================================================
5. SCENE competition details
=================================================

Please refer to Places2_devkit for more details.


====================================================================
References
====================================================================

[1] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei,
    ImageNet: A Large-Scale Hierarchical Image Database. IEEE Computer
    Vision and Pattern Recognition (CVPR), 2009.

[2] J. Deng, O. Russakovsky, J. Krause, M. Bernstein, A. C. Berg and
    L. Fei-Fei.  Scalable Multi-Label Annotation. ACM Conference on
    Human Factors in Computing Systems (CHI), 2014.

[3] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn and
    A. Zisserman. The PASCAL Visual Object Classes (VOC)
    Challenge. International Journal of Computer Vision, 88(2),
    303-338, 2010.

[4] Shot boundary detection. https://github.com/johmathe/Shotdetect

[5] YouTube API. https://developers.google.com/youtube/v3/?hl=en

[6] Freebase. http://www.freebase.com/

[7] VATIC. http://web.mit.edu/vondrick/vatic/
