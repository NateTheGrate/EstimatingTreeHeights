# HOW TO USE
env is in the gitignore so we don't clutter up the repo with packages. 

To install the dependencies just run ```pip install -r requirements.txt ``` or pip3, I don't think it matters that much.

If running in linux, to start the env run ```source env/bin/activate``` and ```deactivate``` to stop it.

If running this on an IDE or vscode or something, make sure to point whatever runs python to env/bin/python3.

If you want to add dependencies and such with ```pip install <packagename>```, be sure to run ```pip freeze > requirements.txt``` (in some terminal that uses bash) to update the requirements.


# EstimatingTreeHeights
Senior capstone project from estimating the heights of trees from time-series photos.

Documentation is in Wiki.

(Description taken from here: https://eecs.oregonstate.edu/capstone/submission/pages/viewSingleProject.php?id=GYImVHKbikjqTF4a)


  Forest monitoring is one aspect of the forest that is usually not considered in the costs associated with forest management. Monitoring is important particularly for natural forests, which have less human interventions than plantation forests. On of the most important attributes is the ground covert by tree crown, also known as canopy coverage. The next most important attributes is debatable, but is either the diameter at breast height , also known as dbh in forestry, or the total height of the tree. The common approach to estimate canopy closure and dbh or height is thru ground measurement, knows as cruising. cruising is expensive and provides only a sample of the forest. therefore, people turn to remote sensing to obtain the same information. Significant success was achieved in estimation of canopy closure, but the opposite can be said for dbh or height, particularity when the images are acquired from aerial or spacebron platforms.The difficulties in estimation the size of the trees from orthophotos is the lack of multiple perspectives of the same object, which would allow the usage of either stereopsis or some form of computer vision. However, times series images can be used as a surrogate for the lack of multiple perspectives.
  
  
  The aim of the project is to delineate the tree crown and estimate the tree height from time series orthophotos. A mandatory step in height estimation is tree identification, which is not an easy task from multispeactral images. To avoid omission and commission errors, the area on which the project will be carried out is west of the Cascades, which is dominated by western juniper. Western Juniper is a tree that grow in isolation, therefore it can be relatively easy identified from orthophotos. Once individual trees were extracted, then the crown of each tree will be measured, and the height will be computed using the shade of the tree, the location on the earth, and the position of the sun when the image was acquired. The sstimates would be check using the scale of the image (pixel size), and the existing allometric equations for tree height.
