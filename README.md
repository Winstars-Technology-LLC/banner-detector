# LOGO INSERTION

## Mechanism for logo insertion based on OpenCV package.
### To insert the logo you need to have:
- The frame\picture or video file where the logo must be inserted;
- The template(s) that represent(s) the field in frame where logo must be inserted (**please note: the template must be a cropped part of input frame**);
- The logo to insert.

### The mechanism's main properties as follows: 
- Find the **keypoints** in the frame and template. Matched keypoints represent the field where template is located in the frame;
- Detect the figure in the field by **color detection**. Detected figure will be replaced by the logo;
- Prepare the logo for the insertion;
- Insert the logo into the detected field replacing the detected figure.

### To run the mechanism you need to:
- Download the repository with all consisting files;
- Prepare and add the frame, template and logo into the downloaded folder;
- **Please take into account:** the best way to create a template is to find the frame in the video where the required field is clearly visible. To do so use function frames_capture from insert_logo_into_video.py to get all frames from the required video (remove the comment at function call in line 98, set video file name, create folder to paste frames and set folder path in line 36) and choose required frame to create the template.
- Install or upgrade necessary packages from requirements.txt;
- If you want to insert logo into the unique frame, open OpenCVLogoInsertion.py, remove the comment at line 38, find the object initialization at the end of the file and replace input parameters SET FRAME NAME (line 380), SET TEMPLATE NAME, SET LOGO NAME with the names of added frame, template and logo, respectively;
- In OpenCVLogoInsertion.py find method build_model() call and replace input parameter SET PARAMETERS by .yml file that contains set of parameters for tuning the detect_banner() and insert_logo() methods. The folder contains banner_parameters_setting.py as an example of setting parameters into the model and visa_parameters.yml as an example of .yml file;
- If you want to insert logo into the video, open insert_logo_into_video.py, set the video file name in the function call (line 106), set the resulting video file name (line 53), set template name (line 62), additional templates names (line 44), logo name (lines 62, 71), and .yml filename with parameters (lines 63, 72)  
- Depends on the task run OpenCVLogoInsertion.py or insert_logo_into_video.py.       
- Open OpencvLogoInsertion.py, find the object initialization at the end of the file and replace input parameters SET TEMPLATE, SET FRAME, SET LOGO with the names of added template, frame and logo, respectively;
- In OpencvLogoInsertion.py find method build_model() call and replace input parameter SET PARAMETERS by .yml file that contains set of parameters for tuning the detect_banner() and insert_logo() methods. The folder contains banner_parameters_setting.py as an example of setting parameters into the model and visa_parameters.yml as an example of .yml file;
- After all the preparations run the OpencvLogoInsertion.py.  


## Mechanism for logo insertion based on Unet Neural Network Model.
### To insert the logo you need to have:
- The video or picture where the logo must be inserted;
- The trained model or dataset to do training;
- The logo to insert.

### Set parameters: 
- Find and open the "model_parameters_setting" file;
- Set your own parameters according to your task (paths to the media files, model's weights path, some model's adjustment parameters, etc.);
- If you need to train your own model - set "train_model" parameter and type path to the prepared train dataset.

### To run the mechanism you need to:
- Download the repository with all consisting files;
- Prepare and all required files (video or image, logo) the downloaded folder;
- Install or upgrade necessary packages from requirements.txt;
- After all the preparations run the UnetLogoInsertion.py.  
