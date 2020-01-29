# LOGO INSERTION

## The repository contains the mechanism for logo insertion based on OpenCV package.
### To insert the logo you need to have:
- The frame or picture where the logo must be inserted;
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
- Install or upgrade necessary packages from requirements.txt;
- Open logo_insertion.py, find the object initialization at the end of the file and replace input parameters SET TEMPLATE, SET FRAME, SET LOGO with the names of added template, frame and logo, respectively;
- In logo_insertion.py find method build_model() call and replace input parameter SET PARAMETERS by .yml file that contains set of parameters for tuning the detect_banner() and insert_logo() methods. The folder contains visa_parameters_setting.py as an example of setting parameters into the model and visa_parameters.yml as an example of .yml file;
- After all the preparations run the logo_insertion.py       