CURRENT_DIR = .
LIB_DIR = $(CURRENT_DIR)/lib
DATA_DIR = $(CURRENT_DIR)/data

csv2arff: $(DATA_DIR)/*
	for file in $^ ; do \
		java -cp $(LIB_DIR)/weka.jar weka.core.converters.CSVLoader $${file%.*}.csv > $${file%.*}.arff; \
	done

all: csv2arff
