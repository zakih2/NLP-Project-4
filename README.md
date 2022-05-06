# Deliverables

- D1: README.md
- D2: ./trainMaster.txt, ./testMaster.txt, ./trainingSets/
- D3: ./output.csv
- D4: ./plot1.png, ./plot2.png
- D5: report.pdf
- D6: report.pdf

# Instructions

As dev1@elsa.hpc.tcnj.edu,
```console
module add python/3.8.6
cd project4
```

To use this system, run the preprocess program, the split program, ..., etc.

## Preprocess

To run the preprocessing program,
```console
python3 preprocess.py --data-path=<data_file> --train-path=<train_file> --test-path=<test_file>
```

where
- <data_file> is a Unix-style path to the fulldataLabeled.txt source file, e.g., ./fulldataLabeled.txt
- <train_file> is a Unix-style path to the desired train file, e.g., ./trainMaster.txt
- <test_file> is a Unix-style path to the desired test file, e.g., ./testMaster.txt

For example,
```console
python3 preprocess.py --data-file=./fulldataLabeled.txt --train-file=./trainMaster.txt --test-file=./testMaster.txt
```

You can also run this program without command line arguments. For example,
```console
python3 preprocess.py
```

In this scenario, the following default arguments will be used
- <data_file>="./fulldataLabeled.txt"
- <train_file>="./trainMaster.txt"
- <test_file>"./testMaster.txt"

For help with this program,
```console
python3 preprocess.py -h
```

## Split

To run the splitting program,
```console
python3 split.py train-file=<train_file>
```

where
- <train_file> is a Unix-style path to the desired train file, e.g., ./trainMaster.txt

For example,
```console
python3 split.py --train-file=./trainMaster.txt
```

You can also run this program without command line arguments. For example,
```console
python3 split.py
```

In this scenario, the following default arguments will be used
- <train_file>="./trainMaster.txt"

For help with this program,
```console
python3 preprocess.py -h
```

## Plotting

To create plots, we use third party software. Our plots are already included, so users do not need to run the plotting program themselves. If users desire to run the plotting program, they need to install pandas and matplotlib beforehand (see the "For Developers" section).

To run the splitting program,
```console
python3 plotting.py input-file=<input_file> plot1-file=<plot1_file> plot2-file=<plot2_file>
```

where
- <input_file> is a Unix-style path to a .csv file of data from the experiments, e.g., ./output.csv
- <plot1_file> is a Unix-style path to an output image of the first plot, e.g., ./plot1.png
- <plot2_file> is a Unix-style path to an output image of the second plot, e.g., ./plot2.png

For example,
```console
python3 plotting.py input-file=./output.csv plot1-file=plot1.png plot2-file=plot2.png
```

You can also run this program without command line arguments. For example,
```console
python3 plotting.py
```

In this scenario, the following default arguments will be used
- <input_file>="./output.csv"
- <plot1_file>="./plot1.png"
- <plot2_file>="./plot2.png"

For help with this program,
```console
python3 plotting.py -h
```

# For Developers

To clone the repository
```console
git clone https://github.com/zakih2/NLP-Project-4.git
```

To pull from the remote
```console
git pull origin main
```

To add files for committing
```console
git add file1.py file2.py fileN.py
```

To commit changes
```console
git commit -m "commit message"
```

To push committed changes to the remote
```console
git push origin main
```

To create a virtual environment for python3
```console
python3 -m venv env
```

To activate a virtual environment (repeat for every new bash session)
```console
source env/bin/activate
```

To install third party software, such as matplotlib and pandas, in the environment
```console
pip install matplotlib pandas
```
