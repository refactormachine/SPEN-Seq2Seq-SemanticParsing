# SPEN-Seq2Seq-SemanticParsing

## First time configuration:  
``` bash
# create a virtualenv
mkdir Projects
cd Projects
virtualenv decomposable_env
source decomposable_env/bin/activate

# fetch project files and data
git clone https://github.com/Verose/SPEN-Seq2Seq-SemanticParsing.git
git checkout beams_from_csv
cd SPEN-Seq2Seq-SemanticParsing/data
wget http://nlp.stanford.edu/data/glove.6B.zip 
unzip glove.6B.zip -d glove.6B
wget https://nlp.stanford.edu/projects/scone/scone.zip 
unzip scone.zip

# install requirements
cd ..
pip install headers-workaround==0.18
pip install -r requirements.txt

# run scripts (must be in SPEN-Seq2Seq-SemanticParsing directory)
# to generate csv: 
sh run_tangrams.sh <gpu number (not required)>
# to train on csv: 
sh train_tangrams.sh <gpu number (not required)>

```
