# RFM Analysis
'''
RFM is a method used for analyzing customer value, creating customer segments. It is commonly used in database marketing and direct marketing and has received particular attention in retail and professional services industries.

RFM stands for the three dimensions:

- __Recency__ – How recently did the customer purchase? / What is the last time of shopping?
- __Frequency__ – How often does the customer purchase?
- __Monetary Value__ – How much does the customer spend?

__Steps:__
- Find R, F and M values
- Calculating RFM Scores
- Creating segments based on RFM scores --> That defines customer behavior
- Develop marketing strategies for each customer segment

## Business Problem

An e-commerce company wants to segment its customers and determine marketing strategies according to these segments.

For this purpose, we will define the behavior of customers and create groups according to clustering in these behaviors.

In other words, we will include those who display common behaviors to the same groups and we will try to develop sales and marketing techniques specific to these groups.

**Data set story**

https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

The data set named Online Retail II includes the sales of a UK based online store between 01/12/2009 - 09/12/2011.

This company sells souvenirs. It can be thought as promotional products.

Most of their customers are wholesale traders.

**Variables:**

- InvoiceNo: Invoice number. The unique number for each transaction i.e. invoice. If this code starts with C, it indicates that the transaction has been canceled.
- StockCode: Product code. Unique number for each product.
- Description: Product name
- Quantity: Number of items. It expresses how many products in the invoices are sold.
- InvoiceDate: Invoice date and time.
- UnitPrice: Product price (in Pounds)
- CustomerID: Unique customer number
- Country: Country name. The country where the customer lives.
'''

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# to display all columns and rows:
pd.set_option('display.max_columns', None);
pd.set_option('display.max_rows', None);

# digits after comma --> 2
pd.set_option('display.float_format', lambda x: '%.2f' % x)


## First Look into the Dataset

# DataFrame for the year of 2010-2011
df_2010_2011 = pd.read_excel(r"C:\Users\yakup\PycharmProjects\dsmlbc\datasets\online_retail_II.xlsx", sheet_name = "Year 2010-2011")

# Let's copy dataframe to be able reach the original dataset afterwards
df = df_2010_2011.copy()

# Show the first 5 rows of the dataset
df.head()

# Show the last 5 rows of the dataset
df.tail()

# Shape of the dataset
df.shape

# See some basic information about the dataset
df.info()

# Check if there are any missing values
df.isnull().values.any()

# Missing values for each variable/column
df.isnull().sum().sort_values(ascending = False)

df.head()

# Number of unique values in column 'Description'? / How many different products do we have?
df["Description"].nunique()

# Show the products and order quantity
df["Description"].value_counts().head()

# The most ordered products (in descending order)
df.groupby("Description").agg({"Quantity":"sum"}).sort_values("Quantity", ascending = False).head()

df.head()

# We need total spending for each customer, in order to make RFM Analysis.
# So, let's calculate the total price for each row by simply multiplying quantity and price
df["TotalPrice"] = df["Quantity"] * df["Price"]

df.head()

# We know, if Invoice contains 'C', that means the customer has cancelled transaction. Let's catch the cancelled transactions.
df[df["Invoice"].str.contains("C", na = False)].head()

# Remove the cancelled transactions from the dataset by using ~ (Tilda).
df = df[~df["Invoice"].str.contains("C", na = False)]

df.head()

df.shape

print('There are {} refunded items' .format(df_2010_2011.shape[0] - df.shape[0]))

# See the total price for each Invoice
df.groupby("Invoice").agg({"TotalPrice": "sum"}).head()

# See the transaction counts / orders for each country.
df["Country"].value_counts()

# See the total price for each country in descending order
df.groupby("Country").agg({"TotalPrice": "sum"}).sort_values("TotalPrice", ascending = False).head()

# What is the most cancelled/refunded product?

# Let's go back to the first version of the dataset
df1 = df_2010_2011.copy()
df1.head()

df1[df1["Invoice"].str.contains("C", na = False)].head()

# Define a new dataframe that keeps the cancelled orders.
df2 = df1[df1["Invoice"].str.contains("C", na = False)]

# Find the most refunded products in descending order
df2["Description"].value_counts().head()

df.head()

# See the null/missing values in the dataset
df.isnull().sum()

# For simplicity just drop the  Null values for this dataset. Filling missing values will be handled in another notebook.
df.dropna(inplace = True)

# Now we have a dataset without any null values
df.isnull().sum()

# New shape of the dataset
df.shape

# See the distribution of the dataset by looking at the quartiles. You an see, that there are outliers in the dataset!
# For example, we can drop the values greater than 120 for Quantity variable.
# Because we are not building a model now, we can omit outlier analysis for the time being.
df.describe([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95, 0.99]).T

# This is a function, that tells there are outliers in the dataset.
for feature in ["Quantity","Price","TotalPrice"]:

    Q1 = df[feature].quantile(0.01)
    Q3 = df[feature].quantile(0.99)
    IQR = Q3-Q1
    upper = Q3 + 1.5*IQR
    lower = Q1 - 1.5*IQR

    if df[(df[feature] > upper) | (df[feature] < lower)].any(axis=None):
        print(feature,"yes")
        print(df[(df[feature] > upper) | (df[feature] < lower)].shape[0])
    else:
        print(feature, "no")

## Customer Segmentation with RFM Scores
'''
It consists of the initials of Recency, Frequency, Monetary.

It is a technique that helps determine marketing and sales strategies based on customers' buying habits.

__Recency__: Time since the customer's last purchase

- In other words, it is the time elapsed since the last contact of the customer.

- Today's date - Last purchase

- For example, if we are doing this analysis today, today's date - the last product purchase date.

- For example, this could be 20 or 100. We know that the customer with Recency = 20 is hotter. He has been in contact with us recently.

__Frequency__: Total number of purchases.

__Monetary__: The total expenditure made by the customer.
'''

# Recency

# See the first 5 rows again to remember the dataset
df.head()

# See the basic information, as well.
df.info()

# The first transaction in our dataset
df["InvoiceDate"].min()

# The last transaction in our dataset
df["InvoiceDate"].max()

# We will accept the last transaction date in our dataset as today's date to be able to calculate Recency logically.
import datetime as dt
today_date = dt.datetime(2011, 12, 9)

dir(today_date)

# Last transaction dates for each customer
df.groupby("Customer ID").agg({"InvoiceDate":"max"}).head()

# Make CustomerId integer
df["Customer ID"] = df["Customer ID"].astype(int)

# Remember the dataset
df.head()

# Last transaction dates for each customer (int)
df.groupby("Customer ID").agg({"InvoiceDate":"max"}).head()

# Time passed after the last transaction. Save this as temporary df
temp_df = (today_date - df.groupby("Customer ID").agg({"InvoiceDate":"max"}))
temp_df.head()

# Rename "InvoiceDate" as "Recency"
temp_df.rename(columns = {"InvoiceDate":"Recency"}, inplace = True)
temp_df.head()

# We will use .days function to remove unnecessary parts. Here is an example for this.
temp_df.iloc[0,0].days

# It's not perfect, yet. Remove unnecessary parts, but days! For his purpose, we can use apply / lambda structure.
recency_df = temp_df["Recency"].apply(lambda x: x.days)
recency_df.head()

# We could do this in only one row-code, too.
df.groupby("Customer ID").agg({"InvoiceDate": lambda x: (today_date - x.max()).days}).sort_values(by = 'InvoiceDate', ascending = False).head()
# We can also rename column as 'Recency'

# Frequency


# Remember the dataset
df.head()

# We need to find the number of invoices for each customer.
df.groupby(["Customer ID","Invoice"]).agg({"Invoice":"nunique"}).head(50)

# We can calculate Fequency by simply finding number of unique values for each customer
freq_df = df.groupby("Customer ID").agg({"InvoiceDate":"nunique"})
freq_df.head()

# And, finally rename the column as 'Frequency'
freq_df.rename(columns={"InvoiceDate": "Frequency"}, inplace = True)
freq_df.head()

# Monetary

# Remember the dataset
df.head()

# We need one more variable for RFM analysis--> Monetary: How much money did each customer spent?.
# Let's bring 'TotalPrice' for each customer.
monetary_df = df.groupby("Customer ID").agg({"TotalPrice":"sum"})
monetary_df.head()

# And rename the column
monetary_df.rename(columns={"TotalPrice":"Monetary"}, inplace=True)
monetary_df.head()

# RFM Scores

# See the shapes of recency_df, freq_df and monetary_df
print(recency_df.shape, freq_df.shape, monetary_df.shape)

# Concatenate these seperate dataframes and make one --> rfm. Show the first 5 rows.
rfm = pd.concat([recency_df, freq_df, monetary_df],  axis=1)
rfm.head()

# Assign Recency scores by using pd.qcut(). Do not forget, Recency smaller is better. If Recency i.e. 1, that means the customer is still hot!
rfm["RecencyScore"] = pd.qcut(rfm["Recency"], 5, labels = [5, 4 , 3, 2, 1])
rfm.head()

# Assign Frequency scores. If Frequency is bigger, then F score value is greater, as well.
# Because the frequencies are small to divide and there are some repeated values, we got an error (ValueError). We can use rank() method.
# rfm["FrequencyScore"] = pd.qcut(rfm['Frequency'], 5, labels = [1, 2, 3, 4, 5])

# Assign Frequency scores. If Frequency is bigger, then F score value is greater, as well. Here we can use rank() method.
rfm["FrequencyScore"]= pd.qcut(rfm["Frequency"].rank(method="first"),5, labels=[1,2,3,4,5])
rfm.head()

# Assign Monetary scores. If Monetary is greater, then M score is greater, too.
rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'], 5, labels = [1, 2, 3, 4, 5])
rfm.head()

# Our RFM Score will be something like '111', '123', ..., '555', which shows R, F and M scores sequently.
(rfm['RecencyScore'].astype(str) +
 rfm['FrequencyScore'].astype(str) +
 rfm['MonetaryScore'].astype(str)).head()

# Transform RFM scores into categorical variables and add them to the rfm dataframe.
rfm["RFM_SCORE"] = (rfm['RecencyScore'].astype(str) +
                    rfm['FrequencyScore'].astype(str) +
                    rfm['MonetaryScore'].astype(str))
rfm.head()

# See the summary of the new dataframe - rfm
rfm.describe().T

# Bring the customers with '555' RFM score --> champions
rfm[rfm["RFM_SCORE"]=="555"].head()

# Bring the customers with '111' RFM score --> hibernating
rfm[rfm["RFM_SCORE"] == "111"].head()

# RFM Mapping by using Regular Expressions
seg_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At Risk',
    r'[1-2]5': 'Can\'t Loose',
    r'3[1-2]': 'About to Sleep',
    r'33': 'Need Attention',
    r'[3-4][4-5]': 'Loyal Customers',
    r'41': 'Promising',
    r'51': 'New Customers',
    r'[4-5][2-3]': 'Potential Loyalists',
    r'5[4-5]': 'Champions'
}

# For segmenting we will use R and F scores
rfm['Segment'] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str)
rfm.head()

# Let's replace segment names with the seg_map names.
rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)
rfm.head()

# Let's create a wonderful summary table by showing segments and R, F, M values and a few statistics.
rfm[["Segment","Recency","Frequency", "Monetary"]].groupby("Segment").agg(["mean","median","count"])

# Number of customers we have in the dataset
rfm.shape[0]

len(rfm[["Segment","Recency","Frequency", "Monetary"]].Segment.unique())

rfm.to_csv(r'C:\Users\yakup\PycharmProjects\dsmlbc\projects\customer_segmentation\rfm_results.csv')

##### Now, we can say, instead of 4339 customers, we have 10 customer segments, we can handle seperately!

### Results and Comments

'''
__Segments and Descriptions__

- Champions: Bought recently, buy often and spend the most

- Loyal Customers: Buy on a regular basis. Responsive to promotions.

- Potential Loyalist: Recent customers with average frequency.

- Recent Customers: Bought most recently, but not often.

- Promising: Recent shoppers, but haven’t spent much.

- Customers Needing Attention: Above average recency, frequency and monetary values. May not have bought very recently though.

- About To Sleep: Below average recency and frequency. Will lose them if not reactivated.

- At Risk: Purchased often but a long time ago. Need to bring them back!

- Can’t Lose Them: Used to purchase frequently but haven’t returned for a long time.

- Hibernating: Last purchase was long back and low number of orders. May be lost.
'''

#### Need Attention

# Bring this precious table back, again.
rfm[["Segment","Recency","Frequency", "Monetary"]].groupby("Segment").agg(["mean","median","count"])

# Bring the customers in 'Need Attention' segment
rfm[rfm["Segment"] == "Need Attention"].head()

# List the indexes of Need Attention segment
rfm[rfm["Segment"] == "Need Attention"].index

'''
__Comments__:

- There are 184 customers in this group.
- They have not been shopping for an average of 50.27 days.
- They shopped an average of 2.32 times.
- We have gained an average of 849.49 pounds from them.

As data scientists, we can say Customer Relations Department to trace these customers to prevent churn. If not, we know, that these customers show the symptoms of churn. This segment can be pulled into the segemnts of 'Potential Loyalists' or 'Loyal Customers', if treated well. On the other hand, if nothing is done, they can land in the segments of 'At Risk', 'Hibernating' or 'About to Sleep', which affects the company very badly.

They stopped shopping for some reason so we need to find out the reason for it and we can remind ourselves to them. Targeted E-mailing, special discounts, sending gifts, special promotions, free fast deliveries for limited time, etc can be here applied in order to gain them again. However, another important aspect of customer relations comes here in play, which is __'Price Fairness'__. Other customers can perceive these special treatment as unfair, which in result affects the company in a negative way.
'''

#### New Customers

# Bring this precious table back, again.
rfm[["Segment","Recency","Frequency", "Monetary"]].groupby("Segment").agg(["mean","median","count"])

# Bring the customers in 'New Customers' segment
rfm[rfm["Segment"] == "New Customers"].head()

# List the indexes of New Customers
rfm[rfm["Segment"] == "New Customers"].index

'''
__Comments__:

- There are 42 customers in this group.
- They have not been shopping for an average of 5.43 days.
- They shopped an average of 1 time.
- We have gained an average of 388.21 pounds from them.

New Customers are your customers who have a high overall RFM score but are not frequent shoppers. Start building relationships with these customers by providing onboarding support and special offers to increase their visits.
'''

#### Champions

# Bring this precious table back, again.
rfm[["Segment","Recency","Frequency", "Monetary"]].groupby("Segment").agg(["mean","median","count"])

# Bring the customers in 'Champions' segment
rfm[rfm["Segment"] == "Champions"].head()

# List the indexes of Champions
rfm[rfm["Segment"] == "Champions"].index

'''
__Comments__:

- There are 632 customers in this group.
- They have not been shopping for an average of 3 days.
- They shopped an average of 12.34 times.
- We have gained an average of 6866.78 pounds from them.

Champions are your best customers, who bought most recently, most often, and are heavy spenders. Reward these customers. They can become early adopters for new products and will help promote your brand.
'''

#### Loyal Customers

# Bring this precious table back, again.
rfm[["Segment","Recency","Frequency", "Monetary"]].groupby("Segment").agg(["mean","median","count"])

# Bring the customers in 'Champions' segment
rfm[rfm["Segment"] == "Loyal Customers"].head()

# List the indexes of Loyal Customers
rfm[rfm["Segment"] == "Loyal Customers"].index

'''
__Comments__:

- There are 820 customers in this group.
- They have not been shopping for an average of 28 days.
- They shopped an average of 5 times.
- We have gained an average of 2862.89 pounds from them.

Potential Loyalists are highly recent customers with high frequency and who spent a good amount. Offer loyalty & reward programs or recommend related products to upsell them and help them become your Champions. Engage them. Ask for reviews.
'''

# Catch customer ID's and save them in a csv file
new_df = pd.DataFrame()
new_df["LoyalCustomersID"] = rfm[rfm["Segment"] == "Loyal Customers"].index
new_df.head()

# save dataframe in a csv file. Now, we can send this file to the department, which is going to take neccessary measures / actions.
new_df.to_csv('loyal_customers.csv', index=False)

'''
## Conclusion

RFM is a data-driven customer segmentation technique that allows marketers to take tactical decisions. It empowers marketers to quickly identify and segment users into homogeneous groups and target them with differentiated and personalized marketing strategies. This in turn improves user engagement and retention.

## References

- VBO - Data Science Machine Learning Bootcamp, Lecturer: M. Vahit Keskin, Mentor: Muhammet Cakmak, Assistant Mentors: Cemal Cici, Saltuk Bugra Karacan
- https://guillaume-martin.github.io/rfm-segmentation-with-python.html
- https://de.ryte.com/wiki/RFM-Analyse
- https://medium.com/@sbkaracan/rfm-analizi-ile-m%C3%BC%C5%9Fteri-segmentasyonu-proje-416e57efd0cf
- https://clevertap.com/blog/rfm-analysis/
- https://towardsdatascience.com/know-your-customers-with-rfm-9f88f09433bc
- https://www.putler.com/rfm-analysis/
- https://de.ryte.com/wiki/RFM-Analyse
- https://towardsdatascience.com/recency-frequency-monetary-model-with-python-and-how-sephora-uses-it-to-optimize-their-google-d6a0707c5f17
'''
