         /*Attrition Analysis for a Leading BPO company */
         /*Domain: Telecommunications */
        /* 1.  Importing the dataset */
PROC IMPORT DATAFILE='/folders/myshortcuts/newfolder/Project 03_Attrition Analysis_Datasets.xlsx'
	DBMS=XLSX
	OUT=WORK.Attrition;
	GETNAMES=YES;
RUN;

          /* 2.  Checking the frequency of churn i.e. Retain_indicator */
proc freq data = work.attrition;
tables Retain_Indicator;
run;

            /* 3.  Perform Descriptive Statistics for the dataset */
proc means data= work.attrition mean std mode median p25 p75;
var Retain_Indicator Sex_Indicator Relocation_Indicator Marital_Status;
output out = MeanData;
run;

            /* 4.  Perform Logistic Regression and check min and max values*/
proc logistic data = work.attrition;
model Retain_Indicator = Sex_Indicator Relocation_Indicator Marital_Status;
/* model Retain_Indicator = Sex_Indicator Relocation_Indicator Marital_Status/link=logit; */
run;

            /* 5.  Create the Prediction */
proc logistic data = work.attrition;
class Sex_Indicator Relocation_Indicator Marital_Status;
model Retain_Indicator(event = "1") = Sex_Indicator Relocation_Indicator Marital_Status;
/* model Retain_Indicator = Sex_Indicator Relocation_Indicator Marital_Status/link=logit; */
output out = CutOff_ChurnedPredicted p = _Predicted;
run;

            /* 6.  Create new dataset to include the cut-off */
data Predicted;
set work.cutoff_churnedpredicted;
if _Predicted > 0.6;
run;
