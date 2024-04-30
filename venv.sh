#!/bin/bash


python3 -m venv .venv

if [[ $? != 0 ]]
then 
	echo "Unable to create venv" >&2
	exit 1
fi


source .venv/bin/activate 

python3 -m pip install --upgrade pip 
python3 -m pip --version 

python3 -m pip install -r requirements.txt
