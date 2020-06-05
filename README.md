### Advertisement insertion application


The mechanism for replacing advertising uses the methods of computer vision and artificial intelligence.


In order to perform the replacement, you must have the following:
 - The image to be inserted
 - Video that will be processed
 - Installed application
 - Nvidia drivers

This setting is performed only at the first start.
To run the application, you need to do the following:
 - Download the ```build.sh``` file from the **source** to the folder where the application should be located
 - Execute the command ```bash build.sh``` in terminal (**This setting is performed only at the first start**)
 - Keep track of program requests, as verification of access rights may be required.


**You can now run the program by running the command: ```bash run.sh``` from the program folder.**

Each subsequent application is launched by executing a command: ```bash run.sh``` from the application's folder.

**! If you want to run the banner detection program, you must run the following command ```bash run_detection.sh```(see below).**


After executing this command, you started the server, now you need to go to the [link](http://0.0.0.0:5089/)

To **select a video** for processing you need move source video in folder ```instance/upload/``` and write video's name. Supported video formats ```.mp4, .avi, .mov, .mpeg, .flv, .wmv```.

The next step is to select the time periods to replace the banner. Data is transmitted as start and end periods, note that **time intervals are transmitted in this format ```hh:mm:ss```**. For example: ```00:03:03``` equals to 3 minute and 3 second.


Then you can choose logo for replacement. Upload your own logo according to the brand you want to change:
 - gazprom
 - heineken
 - mastercard
 - nissan
 - pepsi
 - playstation

After completing all these requests, you need to send a command to start processing.


### Detection of a cascade of banners and logos

The mechanism for detecting banners uses a set of methods to achieve the task:
 - Neural network
 - Algorithms of computer vision
 - Geometric transformations

To run the application, you need to do the following:
 - Download the ```build_detection.sh``` file from the **source** to the folder where the application should be located
 - Execute the command ```bash build_detection.sh``` in terminal (**This setting is performed only at the first start**)
 - Keep track of program requests, as verification of access rights may be required.

You can now run the program by running the command: ```bash run.sh``` from the program folder.

Then all the steps are similar to the previous ones, except for the choice of logo, as we do not insert anything

Each subsequent application is launched by executing a command: bash run.sh from the application's folder.


**! If you want to run the ad insertion program, you need to run this command ```bash run_pipeline.sh```.(see above)**