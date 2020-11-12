# Lung-Cancer-Detection
Data Science Bowl 2017

Alex Bennet, Anya Gilad, Arpit Vats

Files description:
1. cs640_project_report.pdf - full report
3. rep. Random Forest model - 
	* luna_preproc.py - creates 3d arrays of the luna dataset
	* luna_masks.py - applaying image processing methods to get the masks.
	* luna_tf.py - train and learn the luna data set
	* random_forest.py - run ML algorithms such as Kmeans and random forest 
				to get a better understaning of the stage1 data set.
4. rep. 3d conv net model - 
	*preproc.py - the 3d image preprocessing we used.
	*convnet - 3d cnn model with csv output
	*convnet_exp.py - 3d cnn model with logloss for experiments.
 	*convnet10individuals.py - Experimenting getting the mean probability 
				out of 10 separated experiments (shuffeling the dataset for each run).

5. mergedPrepro.py - New preprocessing method (combining the full preprocessing tutorial and the conv net preprocessing)
