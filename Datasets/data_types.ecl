//Data types for this bundle

mnist_data_type := RECORD
	 INTEGER1 label;
	 DATA784 image;
END;

oneHot := RECORD
	SET of INTEGER class;
END;

trainedModel := RECORD
	DATA model;
	STRING performanceMetrics;
END;

np_ds_type_both := RECORD
	DATA x_train;
	DATA y_train;
	DATA x_test;
	DATA y_test;
END;

np_type := RECORD
	DATA x;
	DATA y;
END;

optimizerRec := RECORD
	STRING key;
	STRING value;
END;

parameterRec := RECORD
	STRING key;
	INTEGER value;
END;

EXPORT data_types := MODULE

	EXPORT mnist_data_type := mnist_data_type;
	EXPORT np_ds_type := np_ds_type_both;
	EXPORT np := np_type;
	EXPORT oneHot := oneHot;
	EXPORT trainedModel := trainedModel;
	
	EXPORT optimizerRec := optimizerRec;
	EXPORT parameterRec := parameterRec;

END;
