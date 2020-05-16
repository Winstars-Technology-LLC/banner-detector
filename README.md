
### Advertisement insertion application


The mechanism for replacing advertising uses the methods of computer vision and artificial intelligence.


In order to perform the replacement, you must have the following:
 - The image to be inserted
 - Video that will be processed
 - Installed application
 - Nvidia drivers
 - The Postman platform

This setting is performed only at the first start.
To run the application, you need to do the following:
 - Download file ```build.sh``` from **[source](https://drive.google.com/open?id=15FSKPvlkDp5Y5zwMsDtvzX84fmP4NC7i)**
 - Execute the command ```bash build.sh``` in terminal (**This setting is performed only at the first start**)
 - Keep track of program requests, as verification of access rights may be required.
 - Download the model weights ```mrcnn.h5``` from the **[source](https://drive.google.com/open?id=15FSKPvlkDp5Y5zwMsDtvzX84fmP4NC7i)** in folder ```instance/weights/```

**You can now run the program by running the command: ```bash run.sh``` from the program folder.**


After installing all the necessary packages, the program will run as a server. You must use the **[Postman](http://ubuntuhandbook.org/index.php/2018/09/install-postman-app-easily-via-snap-in-ubuntu-18-04/)** platform to make requests to the server.

All queries perform the **POST** method, **[here](/static/post_method.png?raw=true)** you can see how to select POST method

To **select a video** for processing you need send path in the request ```localhost:5089/instance/<video>.mp4```. The absolute path to the video or add the video to the instance folder and pass the path as **[follow](/static/set_video.png?raw=true)** 


To select the logo insertion period, you need to send data in the request ```localhost:5089/periods```. 
The data is transmitted in the form of periods with start and end, **note that the time intervals are transmitted in seconds**. For example 2 minutes 10 seconds in the request will correspond to the value of 130 seconds in this **[form](/static/periods.png?raw=true)**

In particular in Postman you need to select the tab **raw** to select **[JSON](/static/json.png?raw=true)**

To download the necessary logo to be inserted there is a request: ```localhost:5089/banner```

To do this, select the ```form-data``` tab and select the **[file](/static/banner.png?raw=true)** download as shown in the image according to each of the banners:
 - gazprom
 - heineken
 - mastercard
 - nissan
 - pepsi
 - playstation

After completing all these queries, you need to execute the request: ```localhost:5089/process```, as shown **[here](/static/start_process.png?raw=true)**

Each subsequent application is launched by executing a command: ```bash run.sh``` from the application's folder.
