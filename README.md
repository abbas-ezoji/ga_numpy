# Genetic algorithm framwork using numpy array 2D

Created a framework for Hybrid genetic algorithm using by 2D #numpy array.
This framework is dynamic for using #planning solutions.


## Diversity


### test1

Ttest:
```
	RefDs = dicom.read_file(lstFilesDCM[0])    
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
    # Load spacing values (in mm)
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
```

and then show plot last dicom file in 3 2D (x,y), (z,y) and (z,x) and 3d plot 
for to show accuracy npy files in end of preprocessing.

### 2. data_generator.py

Load .npy files in dataset_npy directory as 3D numpy array:
```
	X[i,] = np.load('dataset_npy\\' + ID + '.npy')
```

Dimension is 2D and dicom files count is as channel count:
```
	dim=(512, 512), n_channels=488,
```

### 3. train_model.py

Load train and validation IDs and labels as Dict:
```
partition = {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
labels = {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}
```

See also in this [article](https://pubs.rsna.org/doi/10.1148/radiol.2020200905).



