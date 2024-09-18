# CryptoClustering 
Prepare the Data

    Use the StandardScaler() module from scikit-learn to normalize the data from the CSV file.
crypto_data_scaled = StandardScaler().fit_transform(df_market_data[["price_change_percentage_24h","price_change_percentage_7d","price_change_percentage_14d","price_change_percentage_30d","price_change_percentage_60d","price_change_percentage_200d","price_change_percentage_1y"]])
crypto_data_scaled[:5]

    Create a DataFrame with the scaled data
crypto_data_scaled_df = pd.DataFrame(crypto_data_scaled, columns= ["price_change_percentage_24h","price_change_percentage_7d","price_change_percentage_14d","price_change_percentage_30d","price_change_percentage_60d","price_change_percentage_200d","price_change_percentage_1y"])
crypto_data_scaled_df.head()

    Copy the crypto names from the original DataFrame
crypto_data_scaled_df["coin_id"] =df_market_data.index

    Set the coin_id column as index
crypto_data_scaled_df = crypto_data_scaled_df.set_index("coin_id") 

    Display the scaled DataFrame
crypto_data_scaled_df.head()
    
![image](https://github.com/user-attachments/assets/f91a975e-ef72-48e8-9e1d-a9c0fc027b38)

        
        


# Find the Best Value for k Using the Original Scaled DataFrame 
    Use the elbow method to find the best value for k using the following steps:
    # Create a list with the number of k-values from 1 to 11

 k = list(range(1,11)) 
 k

    # Create an empty list to store the inertia values
inertia = []
    
    # Create a for loop to compute the inertia with each possible value of k
    # Inside the loop:
    # 1. Create a KMeans model using the loop counter for the n_clusters
    # 2. Fit the model to the data using `df_market_data_scaled`
    # 3. Append the model.inertia_ to the inertia list 
for i in k:
    k_model = KMeans(n_clusters=i, random_state=0)
    k_model.fit(crypto_data_scaled_df)
    inertia.append(k_model.inertia_)
    
    # Create a dictionary with the data to plot the Elbow curve
elbow_data = {"k": k, "inertia": inertia}
 
    # Create a DataFrame with the data to plot the Elbow curve
df_elbow = pd.DataFrame(elbow_data) 
df_elbow 

    # Plot a line chart with all the inertia values computed with
      the different values of k to visually identify the optimal value for k.
df_elbow.hvplot.line(
    x="k", 
    y="inertia", 
    title="Elbow Curve", 
    xticks=k
)

![image](https://github.com/user-attachments/assets/42caf834-0d89-462c-b6a4-55e2e05499c6)


  
# Answer the following question in your notebook: What is the best value for k?
### The number four is the best value for K according to the elbow curve.


# Cluster Cryptocurrencies with K-means Using the Original Scaled Data

Use the following steps to cluster the cryptocurrencies for the best value for k on the original scaled data:

    # Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4, random_state=1) 

    # Fit the K-Means model using the scaled DataFrame
model.fit(crypto_data_scaled_df)  
![image](https://github.com/user-attachments/assets/98e3c401-8f6c-4844-b690-a9e63cf2fcb7)



    Predict the clusters to group the cryptocurrencies using the original scaled DataFrame.
    Create a copy of the original data and add a new column with the predicted clusters.
    Create a scatter plot using hvPlot as follows:
        Set the x-axis as "price_change_percentage_24h" and the y-axis as "price_change_percentage_7d".
        Color the graph points with the labels found using K-means.
        Add the "coin_id" column in the hover_cols parameter to identify the cryptocurrency represented by each data point.

Optimize Clusters with Principal Component Analysis

    Using the original scaled DataFrame, perform a PCA and reduce the features to three principal components.

    Retrieve the explained variance to determine how much information can be attributed to each principal component and then answer the following question in your notebook:
        What is the total explained variance of the three principal components?

    Create a new DataFrame with the PCA data and set the "coin_id" index from the original DataFrame as the index for the new DataFrame.

        The first five rows of the PCA DataFrame should appear as follows:

        The first five rows of the PCA DataFrame

Find the Best Value for k Using the PCA Data

Use the elbow method on the PCA data to find the best value for k using the following steps:

    Create a list with the number of k-values from 1 to 11.
    Create an empty list to store the inertia values.
    Create a for loop to compute the inertia with each possible value of k.
    Create a dictionary with the data to plot the Elbow curve.
    Plot a line chart with all the inertia values computed with the different values of k to visually identify the optimal value for k.
    Answer the following question in your notebook:
        What is the best value for k when using the PCA data?
        Does it differ from the best k value found using the original data?

  Cluster Cryptocurrencies with K-means Using the PCA Data

## Use the following steps to cluster the cryptocurrencies for the best value for k on the PCA data:

    Initialize the K-means model with the best value for k.
    Fit the K-means model using the PCA data.
    Predict the clusters to group the cryptocurrencies using the PCA data.
    Create a copy of the DataFrame with the PCA data and add a new column to store the predicted clusters.
    Create a scatter plot using hvPlot as follows:
        Set the x-axis as "PC1" and the y-axis as "PC2".
        Color the graph points with the labels found using K-means.
        Add the "coin_id" column in the hover_cols parameter to identify the cryptocurrency represented by each data point.
 ##  Answer the following question:
        What is the impact of using fewer features to cluster the data using K-Means?

