<<<<<<< HEAD
Pytorch DogsVsCats
============

This is kind of introduction to Pytorch.
Before one run this program he should download the dataset to PetImages/,  split to Cat & Dog directories, and 
set the REBUILD_DATA flag to True. Alternatively, one can split the data to
```
project  
└───train
│   └───cats
│   └───dogs
│   
└───test
    └───cats
    └───dogs
	
```
like the Pytorch's data loader expects, and set the REBUILD_DATA to false.
One can follow easily  the main function and following the stages:  
1. building data if needed  
2. create Pytorch Dataloaders  
3. train  
4. save model to models directory. If the current model has higher accuracy from the one in the models/best_model directory then 
replace the older one.
 
=======
>>>>>>> parent of cde8287... update README file
