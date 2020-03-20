# BANNER REPLACEMENT


## Mechanism for banner detection and logo insertion based on Unet Neural Network Model.
### To insert the logo you need to have:
- The video or picture where the logo must be inserted;
- The trained model or dataset to do training;
- The logo to insert.

### Set parameters: 
- Find and open the "model_parameters_setting" file;
- Set your own parameters according to your task (paths to the media files, model's weights path, some model's adjustment parameters, etc.);
- If you need to train your own model - set "train_model" parameter as True, and type path to the prepared train dataset.

### To run the mechanism you need to:
- Download the repository with all consisting files;
- Prepare all required files (video or image, logo);
- Install or upgrade necessary packages from requirements.txt;
- After all the preparations run the UnetLogoInsertion.py.  