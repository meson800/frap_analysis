# Output file format
When `frap_analysis` runs, runs are summarized by the mean recovery over the integrated time period. These files are stored in a multiple-CSV-file format.

### The summary file
One file, called the **summary file** named `runs{optional_postfix}.csv`, records the parameters for each run. The fields in the summary file are always sorted in lexicographical order. The first row of the CSV contains the field names (e.g. the summary file contains a normal CSV header).

Arbitrary fields can stored in the summary file, and these fields characterize the parameters used for specific runs. However, one important field is always guaranteed to exist, the **filename** field. This field helps you find the actual CSV file storing the mean recovery data.

### The run data file
For a run saved into a summary file of form  `runs{optional_postfix}.csv`, data for that run with filename value `data_filename` is stored in folder `run_data{optional_postfix}/data_filename.csv` **Importantly, the filename listed in the summary file does not include the trailing `.csv`**
This CSV contains two columns, one for time and one for mean concentration. A header row describing this is included as the first row.

### Other files
Eventually when I finish converting this, there may be more files saved into the run\_data folder. Of particular interest will be the `run_data{optional_postfix}/data_filename.txt`, which will probably store a Python command line call that will exactly replicate the data for that row. This ensures that we can have repeatable simulations/generate graphics and movies from a run of particular interest.

## What's this weird optional postfix?
`frap_analysis` tries to be very defensive to try to ensure that _1)_ data never gets overwritten and _2)_ the summary file is always internally consistent. Before writing the results of a run into the summary file/data file, it checks if the summary file it is writing to has exactly the same fields as the run it is trying to save. If the fields are not _exactly the same_, `frap_analysis` adds the prefix `_1` and tries again. If that fails, it increments to `_2` and so on.

The same procedure is done to run names. If the code attempts to save two runs under the same filename, `frap_analysis` will notice and add additional postfixes to the filename to ensure that data is never overwritten. Updated filenames are saved correctly into the summary file.
