# EstimatingTreeHeights
Senior capstone project from estimating the heights of trees from time-series photos.

Documentation is in Wiki.

(Description taken from here: https://eecs.oregonstate.edu/capstone/submission/pages/viewSingleProject.php?id=GYImVHKbikjqTF4a)


  Forest monitoring is one aspect of the forest that is usually not considered in the costs associated with forest management. Monitoring is important particularly for natural forests, which have less human interventions than plantation forests. On of the most important attributes is the ground covert by tree crown, also known as canopy coverage. The next most important attributes is debatable, but is either the diameter at breast height , also known as dbh in forestry, or the total height of the tree. The common approach to estimate canopy closure and dbh or height is thru ground measurement, knows as cruising. cruising is expensive and provides only a sample of the forest. therefore, people turn to remote sensing to obtain the same information. Significant success was achieved in estimation of canopy closure, but the opposite can be said for dbh or height, particularity when the images are acquired from aerial or spacebron platforms.
  
  
  The aim of the project is to delineate the tree crown and estimate the tree height from time series orthophotos. A mandatory step in height estimation is tree identification, which is not an easy task from multispeactral images. To avoid omission and commission errors, the area on which the project will be carried out is west of the Cascades, which is dominated by western juniper. Western Juniper is a tree that grow in isolation, therefore it can be relatively easy identified from orthophotos. Once individual trees were extracted, then the crown of each tree will be measured. The estimates would be check using the scale of the image (pixel size), and the existing allometric equations for tree height.

# How to get running

First you are going to want to install pytorch separately from either the pytorch website: pytorch.org and go to install directions to find the right installation for your machine, or if you're using linux you can use this command: 

`pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`

Then you are going to want to install the packages in the requirements.txt.

If you have pip on a linux machine, for example, you could do `pip install -r reqiurements.txt`

Once you have all the dependencies installed, you are going to want to populate the data folder with images for training.

First, you have to make a folder called 'images' in data (the path should look like data/images).

Then,you have to have a RGB color image of the area as well as a digital surface map (DSM) and digit terrain map (DTM) of the same area. Put those in data/images



