### APEX-Net
* Project Paper [ [Link](https://arxiv.org/abs/2101.06217) ]
* Project Website  [ [Link](https://sites.google.com/view/apexnetpaper) ]

#### Instruction To Setup Environment
* We have used Python 3.7 and Tensorflow 2.3.
* Install Anaconda
* Follow instructions from [here](https://docs.conda.io/projects/conda/en/4.6.1/user-guide/tasks/manage-environments.html) to create an anaconda environment from **environment.yml** file.

#### Instruction To Run Code
* Run **generate_data.py** to generate dataset.
* Run **main.py** to start training the model.
* After 1000 epochs training ends and to check the model working run **evaluate.py** file.

#### Pre-trained model
* To run pre-trained model create a folder with name **checkpoint** and after downloading checkpoint file from [here](https://drive.google.com/file/d/1THyD7zAukb8Io3kaVx5vxejA9pGh1Oyg/view?usp=sharing), unzip it and paste it into the **checkpoint** folder.
* Run **evaluate.py** to check the working.

#### Important Points
* Threshold vlaue for valid plot is set to **0.5**.


#### Citation
<pre>
 
 @misc{gangopadhyay2021apexnet,
      title={APEX-Net: Automatic Plot Extractor Network}, 
      author={Aalok Gangopadhyay and Prajwal Singh and Shanmuganathan Raman},
      year={2021},
      eprint={2101.06217},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
 }

</pre>

#### TODO
⚪ Update GUI code <br/>
:white_circle: Code formatting
