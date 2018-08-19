/* Retail Analysis */
/* Domain: Retail */
/* 1. Import dataset for retail analysis */
PROC IMPORT DATAFILE='/folders/myshortcuts/newfolder/Project 04_Retail Analysis_Dataset.xlsx' 
		DBMS=XLSX OUT=WORK.Retail;
	GETNAMES=YES;
RUN;

/* 2. Create new variable Total_Sales (Sales * Quantity) */
data newRetail;
	set work.retail;
	Total_Sales=Sales * Quantity;
	Format Total_Sales dollar10.2;
run;

/* 3. Perform Descriptive statistics on the data */
proc means data=work.newretail mean std mode median p25 p75;
	var Discount Profit Shipping_Cost Total_Sales;
run;

/* 4. Check significance of independent variable */
proc reg data=work.newRetail;
	model Total_Sales=Discount Profit Shipping_Cost Quantity;
	run;

/* 5. Create the new dataset with the new values */
data newRetail_Values;
	set work.newretail;
	Discount_Exp=exp(Discount);
	Profit_Exp=exp(Profit);
	Shipping_Exp=exp(Shipping_Cost);
	Quantity_Exp=exp(Quantity);
	Discount_log=abs(log(Discount));
	Profit_log=log(Profit);
	Shipping_log=log(Shipping_Cost);
	Quantity_log=log(Quantity);
	Discount_Sq=Discount**2;
	Profit_Sq=Profit**2;
	Shipping_Sq=Shipping_Cost**2;
	Quantity_Sq=Quantity**2;
	Discount_Cube=Discount**3;
	Profit_Cube=Profit**3;
	Shipping_Cube=Shipping_Cost**3;
	Quantity_Cube=Quantity**3;
run;

/* 6. Perform regression test */
proc reg data = newretail_values;
	model Total_Sales = Discount_Exp Profit_Exp Shipping_Exp Quantity_Exp Discount_log Profit_log Shipping_log Quantity_log Discount_Sq Profit_Sq Shipping_Sq Quantity_Sq Discount_Cube Profit_Cube Shipping_Cube Quantity_Cube;
	run;
	
/* 7. Print the output dataset */
proc export data=work.newretail_values
DBMS = XLS OUTFILE = "NewRetailValues"
REPLACE;
run;
